# -*- coding: utf-8 -*-
"""Portal de solo lectura para seguimiento de verificadores Android (Google Play closed testing).

Seguridad y privacidad:
- Autenticacion propia via env `MEALFIT_VERIFICADORES_ACCOUNTS` ("usuario:clave,usuario2:clave2") +
  cookie HttpOnly firmada `mf_verif_session` (nunca requiere cuenta Bioboros ni admin).
- Fail-closed: sin cuentas configuradas o sin secreto de firma (>=16 chars), responde 503.
- Solo lee cuentas con rango `probador:<uid>` vigente en `app_kv_store` y metricas agregadas
  de uso (nunca expone gasto de IA, plan pagado, alertas, roles admin ni contenido privado).
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Literal, Optional

from fastapi import APIRouter, Body, Cookie, Depends, HTTPException, Request, Response
from fastapi.responses import JSONResponse

import regalos_cuenta as rc
from db import USER_CHAT_THREAD_IDS_SQL, execute_sql_query
from rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/verificadores", tags=["verificadores"])

COOKIE_NAME = "mf_verif_session"
_COOKIE_TTL_S = 7 * 86400
_META_GOOGLE_PLAY = 12
_DIAS_META = 14
_TREINTA_DIAS = "now() - interval '30 days'"
_EN_SUS_HILOS = USER_CHAT_THREAD_IDS_SQL.replace("{uid}", "p.id")

_LOGIN_LIMITER = RateLimiter(max_calls=10, period_seconds=60, scope="verificadores_login")
_LISTA_LIMITER = RateLimiter(max_calls=30, period_seconds=60, scope="verificadores_cuentas")

_SQL_VERIFICADORES = (
    "SELECT p.id::text AS user_id, p.email, p.full_name AS nombre, p.created_at AS alta, "
    "(kv.value ->> 'hasta') AS verificador_hasta, "
    "(kv.value ->> 'alta') AS verificador_desde, "
    "(kv.value ->> 'origen') AS origen, "
    "cm.total AS comidas_total, cm.d30 AS comidas_30d, cm.ultima AS comidas_ultima, "
    "msg.total AS mensajes_coach, msg.ultima AS mensajes_ultima, "
    "ia.escaneos, ia.escaneo_ultima, "
    "pw.ultima AS peso_ultima, ag.ultima AS agua_ultima, "
    "(SELECT count(*) FROM ("
    "SELECT (c.consumed_at AT TIME ZONE 'UTC')::date FROM public.consumed_meals c "
    f"WHERE c.user_id = p.id AND c.consumed_at >= {_TREINTA_DIAS} "
    "UNION SELECT (m.created_at AT TIME ZONE 'UTC')::date FROM public.agent_messages m "
    f"WHERE m.role = 'user' AND {_EN_SUS_HILOS} AND m.created_at >= {_TREINTA_DIAS}"
    ") AS dias) AS dias_activos_30d "
    "FROM public.user_profiles p "
    "INNER JOIN public.app_kv_store kv ON kv.key = 'probador:' || p.id::text "
    "LEFT JOIN LATERAL (SELECT count(*) AS total, "
    f"count(*) FILTER (WHERE c.consumed_at >= {_TREINTA_DIAS}) AS d30, max(c.consumed_at) AS ultima "
    "FROM public.consumed_meals c WHERE c.user_id = p.id) cm ON true "
    "LEFT JOIN LATERAL (SELECT count(*) AS total, max(m.created_at) AS ultima "
    f"FROM public.agent_messages m WHERE m.role = 'user' AND {_EN_SUS_HILOS}) msg ON true "
    "LEFT JOIN LATERAL (SELECT count(*) FILTER (WHERE e.node = 'vision_scan') AS escaneos, "
    "max(e.created_at) FILTER (WHERE e.node = 'vision_scan') AS escaneo_ultima "
    "FROM public.llm_usage_events e WHERE e.user_id = p.id) ia ON true "
    "LEFT JOIN LATERAL (SELECT max(w.created_at) AS ultima "
    "FROM public.weight_log w WHERE w.user_id = p.id) pw ON true "
    "LEFT JOIN LATERAL (SELECT max(a.updated_at) AS ultima "
    "FROM public.water_intake_log a WHERE a.user_id = p.id) ag ON true "
    "WHERE jsonb_typeof(kv.value) = 'object' "
    "AND (kv.value ->> 'hasta') ~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}' "
    "AND (kv.value ->> 'hasta')::timestamptz > now() "
    "ORDER BY GREATEST(cm.ultima, msg.ultima, ia.escaneo_ultima, pw.ultima, ag.ultima) DESC NULLS LAST, "
    "p.created_at DESC, p.id::text "
    "LIMIT 500"
)


def _cuentas_configuradas() -> dict[str, str]:
    crudo = os.environ.get("MEALFIT_VERIFICADORES_ACCOUNTS") or ""
    cuentas: dict[str, str] = {}
    for tramo in crudo.split(","):
        tramo = tramo.strip()
        if ":" not in tramo:
            continue
        usuario, clave = tramo.split(":", 1)
        usuario = usuario.strip().lower()
        clave = clave.strip()
        if usuario and clave:
            cuentas[usuario] = clave
    return cuentas


def _secreto_firma() -> str:
    s = (os.environ.get("MEALFIT_VERIFICADORES_SECRET") or os.environ.get("MEALFIT_SESSION_SECRET") or "").strip()
    return s if len(s) >= 16 else ""


def _configurado() -> bool:
    return bool(_cuentas_configuradas()) and bool(_secreto_firma())


def _b64u_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64u_decode(txt: str) -> bytes:
    pad = "=" * (-len(txt) % 4)
    return base64.urlsafe_b64decode(txt + pad)


def _firmar_token(usuario: str, ahora: Optional[int] = None) -> Optional[str]:
    secreto = _secreto_firma()
    if not secreto:
        return None
    ts = int(ahora if ahora is not None else time.time())
    cuerpo = json.dumps(
        {"typ": "mf_verif", "sub": usuario, "iat": ts, "exp": ts + _COOKIE_TTL_S},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    b64 = _b64u_encode(cuerpo)
    firma = hmac.new(secreto.encode("utf-8"), b64.encode("ascii"), hashlib.sha256).hexdigest()
    return f"{b64}.{firma}"


def _verificar_token(token: Optional[str]) -> Optional[str]:
    if not token or "." not in token:
        return None
    secreto = _secreto_firma()
    cuentas = _cuentas_configuradas()
    if not secreto or not cuentas:
        return None
    try:
        b64, firma = token.split(".", 1)
        esperada = hmac.new(secreto.encode("utf-8"), b64.encode("ascii"), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(firma, esperada):
            return None
        datos = json.loads(_b64u_decode(b64).decode("utf-8"))
        if datos.get("typ") != "mf_verif":
            return None
        exp = int(datos.get("exp") or 0)
        if exp <= int(time.time()):
            return None
        usuario = str(datos.get("sub") or "").strip().lower()
        return usuario if usuario in cuentas else None
    except Exception:                                                  # noqa: BLE001
        return None


def _es_https(request: Request) -> bool:
    proto = (request.headers.get("x-forwarded-proto") or "").split(",")[0].strip().lower()
    return request.url.scheme == "https" or proto == "https"


def _set_cookie(response: Response, request: Request, token: str) -> None:
    response.set_cookie(
        key=COOKIE_NAME,
        value=token,
        max_age=_COOKIE_TTL_S,
        path="/",
        secure=_es_https(request),
        httponly=True,
        samesite="lax",
    )


def _clear_cookie(response: Response, request: Request) -> None:
    response.delete_cookie(
        key=COOKIE_NAME,
        path="/",
        secure=_es_https(request),
        httponly=True,
        samesite="lax",
    )


def _exigir_verificador(
    mf_verif_session: Optional[str] = Cookie(None, alias=COOKIE_NAME),
) -> str:
    if not _configurado():
        raise HTTPException(
            status_code=503,
            detail="Portal de verificadores no configurado en este servidor.",
        )
    usuario = _verificar_token(mf_verif_session)
    if not usuario:
        raise HTTPException(status_code=401, detail="Inicia sesion para ver el seguimiento.")
    return usuario


def _dt_utc(val) -> Optional[datetime]:
    if not val:
        return None
    if isinstance(val, datetime):
        return val if val.tzinfo else val.replace(tzinfo=timezone.utc)
    try:
        dt = datetime.fromisoformat(str(val).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _n(val) -> int:
    try:
        return int(val or 0)
    except (TypeError, ValueError):
        return 0


def _fila_publica(r: dict, ahora: Optional[datetime] = None) -> dict:
    ahora = ahora or datetime.now(timezone.utc)
    marcas = [
        dt for dt in (
            _dt_utc(r.get("comidas_ultima")),
            _dt_utc(r.get("mensajes_ultima")),
            _dt_utc(r.get("escaneo_ultima")),
            _dt_utc(r.get("peso_ultima")),
            _dt_utc(r.get("agua_ultima")),
        )
        if dt is not None
    ]
    ultima = max(marcas) if marcas else None
    hace_7d = ahora - timedelta(days=7)
    dias_activos = _n(r.get("dias_activos_30d"))
    return {
        "email": r.get("email") or "",
        "nombre": r.get("nombre") or None,
        "alta": rc.iso(r.get("alta")),
        "verificador_desde": rc.iso(r.get("verificador_desde")),
        "verificador_hasta": rc.iso(r.get("verificador_hasta")),
        "origen": r.get("origen") or "android",
        "activo_7d": bool(ultima and ultima >= hace_7d),
        "cumple_14d": dias_activos >= _DIAS_META,
        "actividad": {
            "ultima": rc.iso(ultima),
            "dias_activos_30d": dias_activos,
            "comidas_total": _n(r.get("comidas_total")),
            "comidas_30d": _n(r.get("comidas_30d")),
            "mensajes_coach": _n(r.get("mensajes_coach")),
            "escaneos": _n(r.get("escaneos")),
        },
    }


@router.get("/yo")
def verificador_yo(
    mf_verif_session: Optional[str] = Cookie(None, alias=COOKIE_NAME),
):
    cfg = _configurado()
    usuario = _verificar_token(mf_verif_session) if cfg else None
    return {
        "autenticado": bool(usuario),
        "usuario": usuario,
        "configurado": cfg,
    }


@router.post("/login")
def verificador_login(
    request: Request,
    response: Response,
    payload: dict = Body(...),
    _rl: object = Depends(_LOGIN_LIMITER),
):
    cuentas = _cuentas_configuradas()
    if not cuentas or not _secreto_firma():
        return JSONResponse(
            status_code=503,
            content={"detail": "Acceso de colaboradores aun no configurado en el servidor."},
        )
    usuario = str(payload.get("usuario") or "").strip().lower()
    clave = str(payload.get("clave") or "").strip()

    # Recorre todas las cuentas en tiempo constante para evitar enumeracion de usuarios.
    esperada = cuentas.get(usuario, "__dummy_placeholder_key__")
    coincide_clave = hmac.compare_digest(clave.encode("utf-8"), esperada.encode("utf-8"))
    valido = bool(usuario in cuentas and clave and coincide_clave)
    if not valido:
        return JSONResponse(
            status_code=401,
            content={"detail": "Usuario o contrasena incorrectos."},
        )

    token = _firmar_token(usuario)
    if not token:
        return JSONResponse(status_code=503, content={"detail": "No se pudo iniciar sesion."})
    _set_cookie(response, request, token)
    return {"ok": True, "usuario": usuario}


@router.post("/logout")
def verificador_logout(
    request: Request,
    response: Response,
    _rl: object = Depends(_LISTA_LIMITER),
):
    _clear_cookie(response, request)
    return {"ok": True}


@router.get("/cuentas")
def verificador_listar_cuentas(
    buscar: str = "",
    estado: Literal["todos", "activos", "sin_uso"] = "todos",
    _usuario: str = Depends(_exigir_verificador),
    _rl: object = Depends(_LISTA_LIMITER),
):
    try:
        filas_sql = execute_sql_query(_SQL_VERIFICADORES, (), fetch_all=True) or []
    except Exception as e:                                             # noqa: BLE001
        logger.warning(f"[VERIFICADORES] fallo al consultar verificadores: {type(e).__name__}")
        raise HTTPException(status_code=503, detail="No se pudo cargar la lista en este momento.")

    ahora = datetime.now(timezone.utc)
    todas = [_fila_publica(f, ahora=ahora) for f in filas_sql]
    resumen = {
        "total": len(todas),
        "meta_google_play": _META_GOOGLE_PLAY,
        "faltan": max(0, _META_GOOGLE_PLAY - len(todas)),
        "activos_7d": sum(1 for t in todas if t["activo_7d"]),
        "con_actividad": sum(1 for t in todas if t["actividad"]["ultima"] is not None),
        "cumplen_14d": sum(1 for t in todas if t["cumple_14d"]),
        "dias_meta": _DIAS_META,
    }

    q = (buscar or "").strip().lower()
    filtradas = todas
    if q:
        filtradas = [
            t for t in filtradas
            if q in (t["email"] or "").lower() or q in (t["nombre"] or "").lower()
        ]
    if estado == "activos":
        filtradas = [t for t in filtradas if t["activo_7d"]]
    elif estado == "sin_uso":
        filtradas = [t for t in filtradas if not t["activo_7d"]]

    return {"verificadores": filtradas, "resumen": resumen, "total_filtrado": len(filtradas)}
