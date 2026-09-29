# backend/routers/admin.py
"""[P1-PLAN-LOTE-577 · 2026-09-27] Panel de administración, capa 1 (spec 2026-09-27-panel-admin-design §1-§2).

Solo lectura y solo agregados. Las rutas NO llevan «/admin/» en el decorador: ese patrón lo reserva
`test_p2_audit_4_admin_endpoints_token_required` a los endpoints de mantenimiento con CRON_SECRET. Este panel se
protege con la sesión del dueño + la lista `MEALFIT_ADMIN_USER_IDS` (`require_admin`): más estricto que un secreto
compartido, y a cualquier otro le responde 404.
"""
import logging
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel, Field

import admin_cuentas as ac
from admin_acceso import registrar_acceso, require_admin
from admin_metricas import metricas
from rate_limiter import RateLimiter

logger = logging.getLogger(__name__)
# Revisión final: `require_admin` en el PROPIO router — toda ruta que se le añada (la capa 2 traerá contenido de
# usuarios) lo hereda aunque su autor lo olvide. Las rutas lo repiten para recibir el id del admin (FastAPI lo resuelve
# una sola vez por petición).
router = APIRouter(prefix="/api/admin", tags=["admin"], dependencies=[Depends(require_admin)])


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


# [P1-PLAN-LOTE-774 · 2026-09-28] Cuentas: buscar por correo EXACTO, ficha y regalos (spec 2026-09-28-admin-cuentas-
# regalos-design §4.1). Todo POST exige `X-Admin-Accion: 1`: defensa extra además de la cookie SameSite=Strict (un
# formulario de otro sitio no puede poner cabeceras propias sin preflight). La respuesta de cada acción trae la ficha
# nueva, así el panel no necesita otra petición.
_CUENTAS_LECTURA_LIMITER = RateLimiter(max_calls=30, period_seconds=60)
_CUENTAS_ESCRITURA_LIMITER = RateLimiter(max_calls=20, period_seconds=60)


def _exigir_cabecera(x_admin_accion: Optional[str] = Header(None, alias="X-Admin-Accion")) -> None:
    if x_admin_accion != "1":
        raise HTTPException(status_code=403, detail="Falta la cabecera X-Admin-Accion.")


class _Busqueda(BaseModel):
    email: str = Field(max_length=320)


class _Creditos(BaseModel):
    medidor: str
    modo: str
    cantidad: Optional[int] = None
    hasta: str = "mes"
    motivo: str = Field(max_length=400)


class _Cortesia(BaseModel):
    plan: str
    hasta: Optional[str] = None
    motivo: str = Field(max_length=400)


class _Revocar(BaseModel):
    motivo: str = Field(max_length=400)


def _anotar_vista(admin_id: str, accion: str, objetivo, detalle: dict) -> None:
    try:
        registrar_acceso(admin_id, accion, objetivo, detalle)
    except Exception as e:
        logger.error(f"🛑 [P1-PLAN-LOTE-774] no se pudo anotar {accion}: {e!r}")
        raise HTTPException(status_code=503, detail="No se pudo registrar el acceso.")


def _hecho(user_id: str, resultado: dict) -> dict:
    # Revisión final: el cambio YA se guardó. Si releer la ficha falla no es un 500 (invitaría a reintentar y a
    # duplicar el regalo): `cuenta: None` y el panel la vuelve a pedir.
    try:
        cuenta = ac.ficha(user_id)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"⚠️ [P1-PLAN-LOTE-774] cambio guardado pero la ficha de {user_id} no se pudo releer: {e!r}")
        cuenta = None
    return {"ok": True, **resultado, "cuenta": cuenta}


@router.post("/cuentas/buscar", dependencies=[Depends(_exigir_cabecera), Depends(_CUENTAS_LECTURA_LIMITER)])
def api_admin_buscar_cuenta(body: _Busqueda, admin_id: str = Depends(require_admin)):
    uid = ac.buscar_por_correo(body.email)
    _anotar_vista(admin_id, "buscar_cuenta", uid, {"encontrada": bool(uid)})
    return {"cuenta": ac.ficha(uid) if uid else None}


@router.get("/cuentas/{user_id}", dependencies=[Depends(_CUENTAS_LECTURA_LIMITER)])
def api_admin_ver_cuenta(user_id: uuid.UUID, admin_id: str = Depends(require_admin)):
    cuenta = ac.ficha(str(user_id))
    if not cuenta:
        raise HTTPException(status_code=404, detail="No existe esa cuenta.")
    _anotar_vista(admin_id, "ver_cuenta", str(user_id), {})
    return {"cuenta": cuenta}


@router.post("/cuentas/{user_id}/creditos", dependencies=[Depends(_exigir_cabecera), Depends(_CUENTAS_ESCRITURA_LIMITER)])
def api_admin_regalar_creditos(user_id: uuid.UUID, body: _Creditos, admin_id: str = Depends(require_admin)):
    try:
        r = ac.regalar_creditos(admin_id, str(user_id), body.medidor, body.modo, body.cantidad, body.hasta, body.motivo)
    except ac.ErrorRegalo as e:
        raise HTTPException(status_code=e.status, detail=e.detalle)
    return _hecho(str(user_id), r)


@router.post("/cuentas/{user_id}/cortesia", dependencies=[Depends(_exigir_cabecera), Depends(_CUENTAS_ESCRITURA_LIMITER)])
def api_admin_dar_cortesia(user_id: uuid.UUID, body: _Cortesia, admin_id: str = Depends(require_admin)):
    try:
        r = ac.dar_cortesia(admin_id, str(user_id), body.plan, body.hasta, body.motivo)
    except ac.ErrorRegalo as e:
        raise HTTPException(status_code=e.status, detail=e.detalle)
    return _hecho(str(user_id), r)


@router.post("/regalos/{grant_id}/revocar", dependencies=[Depends(_exigir_cabecera), Depends(_CUENTAS_ESCRITURA_LIMITER)])
def api_admin_revocar_regalo(grant_id: uuid.UUID, body: _Revocar, admin_id: str = Depends(require_admin)):
    try:
        r = ac.revocar(admin_id, str(grant_id), body.motivo)
    except ac.ErrorRegalo as e:
        raise HTTPException(status_code=e.status, detail=e.detalle)
    return _hecho(r["user_id"], r)
