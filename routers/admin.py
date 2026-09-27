# backend/routers/admin.py
"""[P1-PLAN-LOTE-577 · 2026-09-27] Panel de administración, capa 1 (spec 2026-09-27-panel-admin-design §1-§2).

Solo lectura y solo agregados. Las rutas NO llevan «/admin/» en el decorador: ese patrón lo reserva
`test_p2_audit_4_admin_endpoints_token_required` a los endpoints de mantenimiento con CRON_SECRET. Este panel se
protege con la sesión del dueño + la lista `MEALFIT_ADMIN_USER_IDS` (`require_admin`): más estricto que un secreto
compartido, y a cualquier otro le responde 404.
"""
import logging

from fastapi import APIRouter, Depends, HTTPException, Query

from admin_acceso import registrar_acceso, require_admin
from admin_metricas import metricas

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/admin", tags=["admin"])


@router.get("/yo")
def api_admin_yo(admin_id: str = Depends(require_admin)):
    try:
        registrar_acceso(admin_id, "abrir_panel")
    except Exception as e:
        logger.error(f"🛑 [P1-PLAN-LOTE-577] no se pudo anotar el acceso al panel: {e!r}")
        raise HTTPException(status_code=503, detail="No se pudo registrar el acceso.")
    return {"ok": True}


@router.get("/metricas")
def api_admin_metricas(dias: int = Query(default=7, ge=1, le=90), admin_id: str = Depends(require_admin)):
    return metricas(dias)
