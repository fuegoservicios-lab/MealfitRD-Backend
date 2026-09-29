# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-848 · 2026-09-29] Sign in with Apple: canjear el código, guardar el refresh token y REVOCARLO al
borrar la cuenta.

Por qué: la guía de borrado de cuentas de Apple (App Review 5.1.1(v)) pide que, si la app ofrece «Continuar con
Apple», al borrar la cuenta se revoquen sus tokens con la API REST de Apple (`/auth/revoke`). Hasta el lote 146 el
binario solo entregaba el identity token (un JWT que se verifica y se tira: `apple_auth.py`), así que no había nada
que revocar. Auditoría App Store 2026-09-29, fila 3.2, §A.6.

Flujo:
  1. El plugin `MfAppleSignIn` devuelve además `authorizationCode` (un solo uso, caduca a los 5 min).
  2. `/api/auth/apple/native`, con la identidad YA verificada y la sesión emitida, lo canjea en `/auth/token` con un
     `client_secret` (JWT ES256 firmado con la clave .p8 de Sign in with Apple) y guarda el `refresh_token` CIFRADO
     (Fernet, clave `MEALFIT_TOKEN_ENC_KEY`) en `public.apple_signin_tokens` — se va con la cuenta por el FK en
     cascada a `user_profiles`. Corre DESPUÉS de responder: el login no espera a Apple ni puede fallar por esto.
  3. `/api/account/delete`, antes de la purga, llama a `/auth/revoke` con ese refresh token. Si falla se anota y el
     borrado sigue: la persona pidió borrar su cuenta y eso no puede depender de que Apple conteste.

Sin configurar (falta la clave .p8, el Team ID, el Key ID o la clave de cifrado): no se canjea ni se revoca, se
anota UNA advertencia y nada se bloquea. Quien entró con Apple antes de este cambio queda cubierto en su siguiente
inicio de sesión.

Nunca se escriben en el log el código, el refresh token, el client_secret ni la clave: solo códigos de estado y el
tipo de excepción. Secretos por `os.environ` (no por `knobs`: `_KNOBS_REGISTRY` guarda el valor crudo y
`get_knobs_registry_snapshot()` lo expone). tooltip-anchor: P1-PLAN-LOTE-848-APPLE-TOKENS
"""
from __future__ import annotations

import logging
import os
import time
from typing import Optional

from knobs import _env_bool, _env_str

logger = logging.getLogger(__name__)

APPLE_TOKEN_URL = "https://appleid.apple.com/auth/token"
APPLE_REVOKE_URL = "https://appleid.apple.com/auth/revoke"
APPLE_AUDIENCE = "https://appleid.apple.com"
# Apple admite un client_secret de hasta 6 meses; se firma uno nuevo por llamada, así que basta con minutos.
CLIENT_SECRET_TTL_S = 300
_HTTP_TIMEOUT_S = 8.0
_HTTP_CONNECT_TIMEOUT_S = 3.0
# Plazo TOTAL de la revocación (lectura de la base + una llamada a Apple) dentro del borrado de cuenta: pasado este
# tiempo el borrado sigue sin esperarla.
REVOKE_DEADLINE_S = 12.0
_MAX_CODE_LEN = 2048
_MAX_TOKEN_LEN = 4096

_avisado_sin_configurar = False


def revocacion_activada() -> bool:
    """Interruptor de emergencia (sin redeploy): `MEALFIT_APPLE_SIWA_TOKENS=false` apaga canje y revocación."""
    return _env_bool("MEALFIT_APPLE_SIWA_TOKENS", True)


def _client_id() -> str:
    return (_env_str("MEALFIT_APPLE_SIWA_CLIENT_ID", "com.bioboros.app") or "com.bioboros.app").strip()


def _clave_privada() -> str:
    """El PEM de la clave .p8: en `APPLE_SIWA_PRIVATE_KEY` (el contenido; admite `\\n` escritos a mano en el .env) o
    en el fichero `APPLE_SIWA_KEY_FILE`."""
    pem = (os.environ.get("APPLE_SIWA_PRIVATE_KEY") or "").strip()
    if pem:
        return pem.replace("\\n", "\n")
    ruta = (os.environ.get("APPLE_SIWA_KEY_FILE") or "").strip()
    if ruta and os.path.isfile(ruta):
        try:
            with open(ruta, encoding="utf-8") as fh:
                return fh.read().strip()
        except OSError as e:
            logger.warning(f"[P1-PLAN-LOTE-848] no se pudo leer APPLE_SIWA_KEY_FILE ({type(e).__name__}).")
    return ""


def _conf() -> dict:
    return {
        "kid": (os.environ.get("APPLE_SIWA_KEY_ID") or "").strip(),
        # El Team ID es el mismo de la cuenta de desarrollador que ya firma la push (APNs): se reutiliza si falta.
        "team": (os.environ.get("APPLE_SIWA_TEAM_ID") or os.environ.get("APNS_TEAM_ID") or "").strip(),
        "clave": _clave_privada(),
        "cifrado": (os.environ.get("MEALFIT_TOKEN_ENC_KEY") or "").strip(),
        "client_id": _client_id(),
    }


def faltantes() -> list:
    """Nombres de lo que falta configurar (nunca valores)."""
    c = _conf()
    falta = []
    if not c["kid"]:
        falta.append("APPLE_SIWA_KEY_ID")
    if not c["team"]:
        falta.append("APPLE_SIWA_TEAM_ID")
    if not c["clave"]:
        falta.append("APPLE_SIWA_PRIVATE_KEY|APPLE_SIWA_KEY_FILE")
    if not c["cifrado"]:
        falta.append("MEALFIT_TOKEN_ENC_KEY")
    else:
        # [ronda 1] Una clave mal formada cuenta como ausente: si no, el canje se hacía (gastando el código de un solo
        # uso) y luego reventaba al cifrar, y el borrado leía la base para nada.
        try:
            from cryptography.fernet import Fernet

            Fernet(c["cifrado"].encode("ascii"))
        except Exception:
            falta.append("MEALFIT_TOKEN_ENC_KEY (mal formada)")
    return falta


def configurado() -> bool:
    return revocacion_activada() and not faltantes()


def _avisar_sin_configurar(donde: str) -> None:
    global _avisado_sin_configurar
    if not revocacion_activada():
        logger.info(f"[P1-PLAN-LOTE-848] {donde}: MEALFIT_APPLE_SIWA_TOKENS=false — sin canje ni revocación.")
        return
    if not _avisado_sin_configurar:
        _avisado_sin_configurar = True
        logger.warning(
            f"[P1-PLAN-LOTE-848] {donde}: Sign in with Apple sin configurar para revocar tokens "
            f"(faltan: {', '.join(faltantes())}). El login y el borrado siguen; la revocación no ocurre."
        )


def construir_client_secret(ahora: Optional[int] = None, client_id: Optional[str] = None) -> str:
    """JWT ES256 que Apple exige como `client_secret` en /auth/token y /auth/revoke.
    iss = Team ID · sub = client_id (bundle) · aud = https://appleid.apple.com · header kid = Key ID."""
    import jwt

    c = _conf()
    t = int(time.time()) if ahora is None else int(ahora)
    return jwt.encode(
        {"iss": c["team"], "iat": t, "exp": t + CLIENT_SECRET_TTL_S, "aud": APPLE_AUDIENCE,
         "sub": client_id or c["client_id"]},
        c["clave"],
        algorithm="ES256",
        headers={"kid": c["kid"]},
    )


def _fernet():
    from cryptography.fernet import Fernet

    return Fernet(_conf()["cifrado"].encode("ascii"))


def cifrar(texto: str) -> str:
    return _fernet().encrypt(texto.encode("utf-8")).decode("ascii")


def descifrar(cifrado: str) -> str:
    return _fernet().decrypt(cifrado.encode("ascii")).decode("utf-8")


def _post_form(url: str, datos: dict):
    """POST x-www-form-urlencoded a Apple. Devuelve (status, json|None). Punto único de red (los tests lo sustituyen)."""
    import httpx

    with httpx.Client(timeout=httpx.Timeout(_HTTP_TIMEOUT_S, connect=_HTTP_CONNECT_TIMEOUT_S)) as cliente:
        r = cliente.post(url, data=datos, headers={"Accept": "application/json"})
    try:
        cuerpo = r.json()
    except Exception:
        cuerpo = None
    return r.status_code, cuerpo


def _sub_del_id_token(id_token) -> Optional[str]:
    """El `sub` del id_token que Apple devuelve en el canje. Sin verificar la firma, a sabiendas: llega por TLS desde
    appleid.apple.com como respuesta a NUESTRO client_secret firmado, no del cliente. Sí se exigen iss y aud."""
    import jwt

    try:
        claims = jwt.decode(
            str(id_token or ""),
            options={"verify_signature": False, "verify_exp": False, "verify_aud": False},
        )
    except Exception:
        return None
    if claims.get("iss") != APPLE_AUDIENCE:
        return None
    aud = claims.get("aud")
    auds = aud if isinstance(aud, list) else [aud]
    if _conf()["client_id"] not in auds:
        return None
    return str(claims.get("sub") or "").strip() or None


def canjear_codigo(codigo: str, sub_esperado: Optional[str] = None) -> Optional[str]:
    """`authorizationCode` → `refresh_token`, o None. Nunca lanza.

    [ronda 1] El código lo manda el cliente junto al identity token, pero nada los ataba: un código de OTRA cuenta de
    Apple se habría guardado en esta. Se exige que el `sub` del id_token del canje sea el de la identidad ya verificada;
    si no coincide (o falta), no se guarda nada."""
    try:
        status, cuerpo = _post_form(APPLE_TOKEN_URL, {
            "client_id": _conf()["client_id"],
            "client_secret": construir_client_secret(),
            "code": codigo,
            "grant_type": "authorization_code",
        })
    except Exception as e:
        logger.warning(f"[P1-PLAN-LOTE-848] canje del código de Apple falló: {type(e).__name__}")
        return None
    if status != 200 or not isinstance(cuerpo, dict):
        error = cuerpo.get("error") if isinstance(cuerpo, dict) else None
        logger.warning(f"[P1-PLAN-LOTE-848] Apple rechazó el canje del código: HTTP {status} ({str(error or '')[:40]})")
        return None
    refresco = str(cuerpo.get("refresh_token") or "").strip()
    if not refresco or len(refresco) > _MAX_TOKEN_LEN:
        logger.warning("[P1-PLAN-LOTE-848] el canje de Apple no trajo refresh_token.")
        return None
    sub = _sub_del_id_token(cuerpo.get("id_token"))
    if not sub_esperado or sub != sub_esperado:
        logger.warning("[P1-PLAN-LOTE-848] el código de Apple no es de la identidad verificada — no se guarda nada.")
        return None
    return refresco


def guardar_refresh_token(user_id: str, refresh_token: str) -> bool:
    from db import execute_sql_write

    return bool(execute_sql_write(
        "INSERT INTO public.apple_signin_tokens (user_id, refresh_token_enc, client_id) VALUES (%s, %s, %s) "
        "ON CONFLICT (user_id) DO UPDATE SET refresh_token_enc = EXCLUDED.refresh_token_enc, "
        "client_id = EXCLUDED.client_id, updated_at = now()",
        (user_id, cifrar(refresh_token), _conf()["client_id"]),
    ))


def canjear_y_guardar(user_id: str, codigo: Optional[str], sub_apple: Optional[str] = None) -> bool:
    """Lo que corre tras el login con Apple. Best-effort: devuelve si quedó guardado y JAMÁS lanza.
    `sub_apple` = el `sub` del identity token YA verificado: el código solo se guarda si es de esa identidad."""
    try:
        codigo = str(codigo or "").strip()
        if not user_id or not codigo or not sub_apple:
            return False
        if len(codigo) > _MAX_CODE_LEN:
            logger.info("[P1-PLAN-LOTE-848] authorizationCode demasiado largo — ignorado.")
            return False
        if not configurado():
            _avisar_sin_configurar("login")
            return False
        refresco = canjear_codigo(codigo, sub_apple)
        if not refresco:
            return False
        guardado = guardar_refresh_token(user_id, refresco)
        logger.info(f"[P1-PLAN-LOTE-848] refresh token de Apple guardado (uid={str(user_id)[:8]}…, ok={guardado}).")
        return guardado
    except Exception as e:
        logger.warning(f"[P1-PLAN-LOTE-848] canjear_y_guardar lanzó {type(e).__name__} (el login no se entera).")
        return False


def revocar_de_usuario(user_id: str) -> dict:
    """Antes de borrar la cuenta: revoca en Apple el refresh token guardado. Best-effort, JAMÁS lanza.
    Devuelve `{"revocado": bool, "motivo": str}` (motivo: ok | sin_token | sin_configurar | http_<n> | error)."""
    try:
        if not configurado():
            _avisar_sin_configurar("borrado de cuenta")
            return {"revocado": False, "motivo": "sin_configurar"}
        from db import execute_sql_query

        fila = execute_sql_query(
            "SELECT refresh_token_enc, client_id FROM public.apple_signin_tokens WHERE user_id = %s",
            (user_id,), fetch_one=True,
        )
        cifrado = (fila or {}).get("refresh_token_enc")
        if not cifrado:
            return {"revocado": False, "motivo": "sin_token"}
        # [ronda 1] El client_id con el que se EMITIÓ ese token (el de la fila), no el configurado hoy: Apple solo
        # revoca un token con el client_id que lo obtuvo.
        client_id = str((fila or {}).get("client_id") or "").strip() or _conf()["client_id"]
        status, _ = _post_form(APPLE_REVOKE_URL, {
            "client_id": client_id,
            "client_secret": construir_client_secret(client_id=client_id),
            "token": descifrar(cifrado),
            "token_type_hint": "refresh_token",
        })
        if status == 200:
            logger.info(f"[P1-PLAN-LOTE-848] tokens de Apple revocados (uid={str(user_id)[:8]}…).")
            return {"revocado": True, "motivo": "ok"}
        logger.warning(f"[P1-PLAN-LOTE-848] Apple no revocó los tokens: HTTP {status} (el borrado sigue).")
        return {"revocado": False, "motivo": f"http_{status}"}
    except Exception as e:
        logger.warning(f"[P1-PLAN-LOTE-848] revocar_de_usuario lanzó {type(e).__name__} (el borrado sigue).")
        return {"revocado": False, "motivo": "error"}


async def revocar_con_plazo(user_id: str, plazo_s: Optional[float] = None) -> dict:
    """[ronda 1] `revocar_de_usuario` con un plazo TOTAL para el borrado de cuenta: si Apple o la base tardan, el
    borrado sigue sin esperar (el hilo termina solo, con sus timeouts de httpx). Nunca lanza."""
    import asyncio

    try:
        return await asyncio.wait_for(
            asyncio.to_thread(revocar_de_usuario, user_id),
            timeout=REVOKE_DEADLINE_S if plazo_s is None else plazo_s,
        )
    except asyncio.TimeoutError:
        logger.warning("[P1-PLAN-LOTE-848] la revocación en Apple pasó el plazo (el borrado sigue).")
        return {"revocado": False, "motivo": "plazo"}
    except Exception as e:
        logger.warning(f"[P1-PLAN-LOTE-848] revocar_con_plazo lanzó {type(e).__name__} (el borrado sigue).")
        return {"revocado": False, "motivo": "error"}
