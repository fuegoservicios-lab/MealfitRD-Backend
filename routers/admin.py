# backend/routers/admin.py
"""[P1-PLAN-LOTE-577 · 2026-09-27] Panel de administración, capa 1 (spec 2026-09-27-panel-admin-design §1-§2).

Solo lectura y solo agregados. Las rutas NO llevan «/admin/» en el decorador: ese patrón lo reserva
`test_p2_audit_4_admin_endpoints_token_required` a los endpoints de mantenimiento con CRON_SECRET. Este panel se
protege con la sesión del dueño + la lista `MEALFIT_ADMIN_USER_IDS` (`require_admin`): más estricto que un secreto
compartido, y a cualquier otro le responde 404.
"""
import logging
import uuid
from datetime import datetime, timezone
from typing import List, Literal, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field

import admin_cuentas as ac
import admin_cuentas_lista as acl
import admin_prueba_detalle as apd
import ajustes_cuenta
import cuentas_prueba as cp
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
    cuenta = ac.ficha(uid) if uid else None
    if cuenta and cp.activo():           # [P1-PLAN-LOTE-831] la ficha ampliada, solo con el interruptor maestro
        cuenta = acl.ampliar_ficha(cuenta)
    return {"cuenta": cuenta}


@router.get("/cuentas/{user_id}", dependencies=[Depends(_CUENTAS_LECTURA_LIMITER)])
def api_admin_ver_cuenta(user_id: uuid.UUID, admin_id: str = Depends(require_admin)):
    cuenta = ac.ficha(str(user_id))
    if not cuenta:
        raise HTTPException(status_code=404, detail="No existe esa cuenta.")
    if cp.activo():                      # [P1-PLAN-LOTE-831] apagado, la ficha es exactamente la del lote 774
        cuenta = acl.ampliar_ficha(cuenta)
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


# [P1-PLAN-LOTE-831 · 2026-09-29] Cuentas con actividad EN NÚMEROS, ajustes y cuentas de prueba (spec
# 2026-09-29-admin-cuentas-actividad-pruebas-design §4.1-§4.3, §13; contratos 1-8 del plan). Todo detrás del interruptor
# maestro `MEALFIT_ADMIN_TEST_ACCOUNTS`: apagado, cada ruta nueva responde el MISMO 404 que a un extraño (no se anuncia)
# antes de gastar cupo, y la ficha de arriba queda exactamente como la del lote 774. Las reglas de la marca viven en
# `cuentas_prueba` (su rastro va ANTES de escribir); la lista y la ficha, en `admin_cuentas_lista`; los ajustes, en
# `ajustes_cuenta`. Toda vista con datos de UNA persona anota su fila antes de responder (fail-closed ⇒ 503).
# El lote va en `/pruebas/lote`, fuera de `/cuentas/…`: `/cuentas/{user_id}/…` lo capturaría y daría 422.
# Cupo propio para la lista y el CSV con un par (max, periodo) que no usa nadie más: Redis comparte la ventana por par
# (`rl:<max>:<periodo>:<uid>`). El contrato decía 40/60, que ya usan otros tres limitadores (la voz del coach entre
# ellos): con ese par el panel y el chat del dueño se comerían el cupo el uno al otro.
_CUENTAS_LISTA_LIMITER = RateLimiter(max_calls=45, period_seconds=60)
_Orden = Literal["actividad", "alta", "comidas", "gasto"]
_Filtro = Literal["todas", "prueba", "sin_marcar", "activas_7d", "inactivas_14d", "con_plan", "seguimiento"]


def _exigir_knob_pruebas() -> None:
    if not cp.activo():
        raise HTTPException(status_code=404, detail="Not Found")


class _MarcaPrueba(BaseModel):
    motivo: str = Field(max_length=400)
    confirmar_vuelta: bool = False


class _QuitarPrueba(BaseModel):
    motivo: str = Field(max_length=400)


class _LotePrueba(BaseModel):
    # El tope real (100) y su código (`demasiadas`) los pone `cuentas_prueba.marcar_varias`; este solo acota el cuerpo.
    user_ids: List[str] = Field(default_factory=list, max_length=1000)
    motivo: str = Field(max_length=400)


def _con_ficha(user_id: str) -> dict:
    """Como `_hecho`, con la ficha AMPLIADA: la marca YA se guardó, así que releerla mal no es un 500 (invitaría a
    reintentar): `cuenta: None` y el panel la vuelve a pedir."""
    try:
        cuenta = acl.ampliar_ficha(ac.ficha(user_id))
    except Exception as e:  # noqa: BLE001
        logger.warning(f"⚠️ [P1-PLAN-LOTE-831] marca guardada pero la ficha de {user_id} no se pudo releer: {e!r}")
        cuenta = None
    return {"ok": True, "cuenta": cuenta}


def _sin_datos(que: str, e: Exception) -> HTTPException:
    logger.error(f"🛑 [P1-PLAN-LOTE-831] {que} no se pudo leer: {e!r}")
    return HTTPException(status_code=503, detail=f"No se pudo leer {que}.")


@router.get("/cuentas", dependencies=[Depends(_exigir_knob_pruebas), Depends(_CUENTAS_LISTA_LIMITER)])
def api_admin_listar_cuentas(buscar: str = Query(default="", max_length=acl.MAX_BUSQUEDA),
                             orden: _Orden = Query(default="actividad"), filtro: _Filtro = Query(default="todas"),
                             pagina: int = Query(default=1, ge=1, le=10000), admin_id: str = Depends(require_admin)):
    try:
        r = acl.listar(buscar, orden, filtro, pagina)
    except Exception as e:  # noqa: BLE001
        raise _sin_datos("la lista de cuentas", e) from e
    _anotar_vista(admin_id, "listar_cuentas", None,
                  {"buscar": buscar, "orden": orden, "filtro": filtro, "pagina": pagina, "n": len(r["cuentas"])})
    return r


@router.get("/cuentas.csv", dependencies=[Depends(_exigir_knob_pruebas), Depends(_CUENTAS_LISTA_LIMITER)])
def api_admin_exportar_cuentas(buscar: str = Query(default="", max_length=acl.MAX_BUSQUEDA),
                               orden: _Orden = Query(default="actividad"), filtro: _Filtro = Query(default="todas"),
                               admin_id: str = Depends(require_admin)):
    try:
        texto, n = acl.exportar_csv(buscar, orden, filtro)
    except Exception as e:  # noqa: BLE001
        raise _sin_datos("el CSV de cuentas", e) from e
    # `buscar` además del contrato ({filtro, n}): sin él, el rastro no diría QUÉ cuentas salieron en el fichero.
    _anotar_vista(admin_id, "exportar_cuentas", None, {"filtro": filtro, "buscar": buscar, "n": n})
    nombre = f"cuentas-{datetime.now(timezone.utc):%Y%m%d}.csv"
    return Response(content=texto.encode("utf-8"), media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="{nombre}"', "Cache-Control": "no-store"})


@router.post("/cuentas/{user_id}/prueba", dependencies=[
    Depends(_exigir_knob_pruebas), Depends(_exigir_cabecera), Depends(_CUENTAS_ESCRITURA_LIMITER)])
def api_admin_marcar_prueba(user_id: uuid.UUID, body: _MarcaPrueba, admin_id: str = Depends(require_admin)):
    try:
        cp.marcar(admin_id, str(user_id), body.motivo, body.confirmar_vuelta)
    except cp.ErrorPrueba as e:
        raise HTTPException(status_code=e.status, detail=e.detalle)
    return _con_ficha(str(user_id))


@router.post("/cuentas/{user_id}/prueba/quitar", dependencies=[
    Depends(_exigir_knob_pruebas), Depends(_exigir_cabecera), Depends(_CUENTAS_ESCRITURA_LIMITER)])
def api_admin_quitar_prueba(user_id: uuid.UUID, body: _QuitarPrueba, admin_id: str = Depends(require_admin)):
    try:
        cp.quitar(admin_id, str(user_id), body.motivo)
    except cp.ErrorPrueba as e:
        raise HTTPException(status_code=e.status, detail=e.detalle)
    return _con_ficha(str(user_id))


@router.post("/pruebas/lote", dependencies=[
    Depends(_exigir_knob_pruebas), Depends(_exigir_cabecera), Depends(_CUENTAS_ESCRITURA_LIMITER)])
def api_admin_marcar_pruebas_en_lote(body: _LotePrueba, admin_id: str = Depends(require_admin)):
    try:
        resultados = cp.marcar_varias(admin_id, body.user_ids, body.motivo)
    except cp.ErrorPrueba as e:
        raise HTTPException(status_code=e.status, detail=e.detalle)
    return {"ok": True, "resultados": resultados}


@router.get("/cuentas/{user_id}/ajustes/historial",
            dependencies=[Depends(_exigir_knob_pruebas), Depends(_CUENTAS_LECTURA_LIMITER)])
def api_admin_historial_ajustes(user_id: uuid.UUID, dias: int = Query(default=90, ge=1, le=365),
                                admin_id: str = Depends(require_admin)):
    try:
        cambios = ajustes_cuenta.historial(str(user_id), dias)
    except Exception as e:  # noqa: BLE001
        raise _sin_datos("el historial de ajustes", e) from e
    _anotar_vista(admin_id, "ver_ajustes", str(user_id), {"dias": dias, "n": len(cambios)})
    return {"cambios": cambios}


@router.get("/ajustes/resumen", dependencies=[Depends(_exigir_knob_pruebas), Depends(_CUENTAS_LECTURA_LIMITER)])
def api_admin_resumen_ajustes(dias: int = Query(default=30, ge=1, le=90), admin_id: str = Depends(require_admin)):
    """Agregado sin las cuentas admin (`ajustes_cuenta.resumen`): no abre ninguna cuenta, así que no deja rastro."""
    try:
        return ajustes_cuenta.resumen(dias)
    except Exception as e:  # noqa: BLE001
        raise _sin_datos("el resumen de ajustes", e) from e


# [P1-PLAN-LOTE-832 · 2026-09-29] El detalle de una cuenta de PRUEBA (spec §4.4, §8 y §13.4; contrato 9): su formulario,
# sus comidas, sus planes, sus conversaciones con sus fotos y su línea de tiempo. Solo lectura. Cada vista, en este
# orden: interruptor (404, antes del cupo) → parámetros (422, sin mirar nada) → `exigir_prueba` en CADA petición, sin
# caché (403 `no_es_prueba`, 409 `aviso_pendiente`: ninguno deja rastro) → fila `ver_prueba` {seccion, objeto} (si no
# se anota, 503 sin datos) → la lectura (`admin_prueba_detalle`; si falla, 503 sin datos). Un plan, un hilo o una foto
# de OTRA cuenta responde 404 aunque exista: la pertenencia la decide la consulta del módulo, y el intento queda en el
# rastro. Cupo propio con el par del contrato (90, 60), que no usa nadie más: un hilo con fotos pide cada foto aparte.
_PRUEBA_DETALLE_LIMITER = RateLimiter(max_calls=90, period_seconds=60)
_DETALLE_DE_PRUEBA = [Depends(_exigir_knob_pruebas), Depends(_PRUEBA_DETALLE_LIMITER)]


def _abrir_prueba(admin_id: str, user_id: str, seccion: str, objeto: Optional[str] = None) -> None:
    """La marca viva (sin caché) y, solo entonces, la fila del rastro ANTES de leer nada."""
    try:
        cp.exigir_prueba(user_id)
    except cp.ErrorPrueba as e:
        raise HTTPException(status_code=e.status, detail=e.detalle)
    _anotar_vista(admin_id, "ver_prueba", user_id, {"seccion": seccion, "objeto": objeto})


def _parametro(validar, *args):
    try:
        return validar(*args)
    except apd.ErrorDetalle as e:
        raise HTTPException(status_code=e.status, detail=e.detalle)


def _leer(que: str, leer, *args):
    try:
        return leer(*args)
    except Exception as e:  # noqa: BLE001
        raise _sin_datos(que, e) from e


@router.get("/cuentas/{user_id}/prueba/formulario", dependencies=_DETALLE_DE_PRUEBA)
def api_admin_prueba_formulario(user_id: uuid.UUID, admin_id: str = Depends(require_admin)):
    uid = str(user_id)
    _abrir_prueba(admin_id, uid, "formulario")
    r = _leer("el formulario", apd.formulario, uid)
    if r is None:
        raise HTTPException(status_code=404, detail="No existe esa cuenta.")
    return r


@router.get("/cuentas/{user_id}/prueba/comidas", dependencies=_DETALLE_DE_PRUEBA)
def api_admin_prueba_comidas(user_id: uuid.UUID, desde: Optional[str] = Query(default=None, max_length=10),
                             hasta: Optional[str] = Query(default=None, max_length=10),
                             admin_id: str = Depends(require_admin)):
    d1, d2 = _parametro(apd.rango_de_comidas, desde, hasta)
    uid = str(user_id)
    _abrir_prueba(admin_id, uid, "comidas", apd.objeto_de_comidas(d1, d2))
    return _leer("las comidas", apd.comidas, uid, d1, d2)


@router.get("/cuentas/{user_id}/prueba/planes", dependencies=_DETALLE_DE_PRUEBA)
def api_admin_prueba_planes(user_id: uuid.UUID, admin_id: str = Depends(require_admin)):
    uid = str(user_id)
    _abrir_prueba(admin_id, uid, "planes")
    return _leer("los planes", apd.planes, uid)


@router.get("/cuentas/{user_id}/prueba/planes/{plan_id}", dependencies=_DETALLE_DE_PRUEBA)
def api_admin_prueba_plan(user_id: uuid.UUID, plan_id: uuid.UUID, admin_id: str = Depends(require_admin)):
    uid, pid = str(user_id), str(plan_id)
    _abrir_prueba(admin_id, uid, "plan", pid)
    r = _leer("el plan", apd.plan, uid, pid)
    if r is None:
        raise HTTPException(status_code=404, detail="No existe ese plan en esta cuenta.")
    return r


@router.get("/cuentas/{user_id}/prueba/conversaciones", dependencies=_DETALLE_DE_PRUEBA)
def api_admin_prueba_conversaciones(user_id: uuid.UUID, admin_id: str = Depends(require_admin)):
    uid = str(user_id)
    _abrir_prueba(admin_id, uid, "conversaciones")
    return _leer("las conversaciones", apd.conversaciones, uid)


@router.get("/cuentas/{user_id}/prueba/conversaciones/{session_id}", dependencies=_DETALLE_DE_PRUEBA)
def api_admin_prueba_conversacion(user_id: uuid.UUID, session_id: uuid.UUID, admin_id: str = Depends(require_admin)):
    uid, sid = str(user_id), str(session_id)
    _abrir_prueba(admin_id, uid, "conversacion", sid)
    r = _leer("la conversación", apd.conversacion, uid, sid)
    if r is None:
        raise HTTPException(status_code=404, detail="No existe esa conversación en esta cuenta.")
    return r


@router.get("/cuentas/{user_id}/prueba/adjuntos/{attachment_id}", dependencies=_DETALLE_DE_PRUEBA)
def api_admin_prueba_adjunto(user_id: uuid.UUID, attachment_id: uuid.UUID, admin_id: str = Depends(require_admin)):
    uid, aid = str(user_id), str(attachment_id)
    _abrir_prueba(admin_id, uid, "adjunto", aid)
    r = _leer("la foto", apd.adjunto, uid, aid)
    if r is None:
        raise HTTPException(status_code=404, detail="No existe esa foto en esta cuenta.")
    contenido, tipo = r
    return Response(content=contenido, media_type=tipo, headers={
        "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff", "Content-Disposition": "inline"})


@router.get("/cuentas/{user_id}/prueba/actividad", dependencies=_DETALLE_DE_PRUEBA)
def api_admin_prueba_actividad(user_id: uuid.UUID, dias: int = Query(default=7, ge=1, le=apd.MAX_DIAS_ACTIVIDAD),
                               tipos: str = Query(default="", max_length=200), admin_id: str = Depends(require_admin)):
    elegidos = _parametro(apd.tipos_de_actividad, tipos)
    uid = str(user_id)
    _abrir_prueba(admin_id, uid, "actividad", apd.objeto_de_actividad(dias, elegidos))
    return _leer("la actividad", apd.actividad, uid, dias, elegidos)
