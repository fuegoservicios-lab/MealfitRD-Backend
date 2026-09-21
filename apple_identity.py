"""[P1-PLAN-LOTE-146 · 2026-09-20] De un `sub` de Apple YA VERIFICADO a un usuario de Bioboros.

La identidad de Bioboros vive en `neon_auth."user"` (Better Auth gestionado por Neon) y el backend
RECHAZA a quien no tenga fila ahí (`P1-AUTH-CUENTA-BORRADA`). Neon Auth no ofrece Apple como
proveedor, así que el enlace lo escribimos nosotros — pero en SUS tablas y con SU forma: una fila en
`neon_auth.account` con `providerId='apple'` y `accountId=<sub>`, exactamente lo que Better Auth
guarda para un proveedor social. Sin tablas propias ni DDL, y la purga de cuenta ya lo limpia (el
DELETE de `user` arrastra `account` en cascada). Esquema medido en producción el 2026-09-20:

    user     id uuid DEFAULT · name NOT NULL · email NOT NULL · "emailVerified" NOT NULL · banned NULL
    account  "accountId" NOT NULL · "providerId" NOT NULL · "userId" uuid NOT NULL · "updatedAt" NOT NULL (sin default)

Tres caminos, en este orden:
  1. el `sub` ya está enlazado            → ese usuario;
  2. Apple da un correo VERIFICADO que ya tiene cuenta → se enlaza a ESA cuenta (la de su OTP/Google);
  3. nadie con ese correo                 → nace la identidad (con el correo-relay si eligió ocultarlo).

Lo que NUNCA pasa: enlazar por un correo que Apple no declara verificado (sería tomar una cuenta ajena
con solo conocer su correo), ni devolver un usuario vetado (`banned`)."""
from __future__ import annotations

import logging
import re
from typing import Optional

from db import execute_sql_query, execute_sql_write

logger = logging.getLogger(__name__)

PROVIDER = "apple"


class AppleIdentityError(Exception):
    """`code` es estable: el router lo traduce a HTTP y el cliente a un mensaje."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def _nombre(name: Optional[str], email: str, is_private: bool) -> str:
    limpio = re.sub(r"[\x00-\x1f<>]", "", str(name or "")).strip()[:80]
    if limpio:
        return limpio
    if not is_private and "@" in email:
        return email.split("@", 1)[0][:80] or "Usuario"
    return "Usuario"


def _vetado(fila: dict) -> bool:
    return bool(fila.get("banned"))


def _enlazar(uid: str, sub: str) -> None:
    execute_sql_write(
        'INSERT INTO neon_auth.account ("accountId", "providerId", "userId", "updatedAt") '
        "VALUES (%s, %s, %s, CURRENT_TIMESTAMP)",
        (sub, PROVIDER, uid),
    )


def _por_sub(sub: str) -> Optional[dict]:
    filas = execute_sql_query(
        'SELECT u.id::text AS id, u.email, u.name, u.banned FROM neon_auth.account a '
        'JOIN neon_auth."user" u ON u.id = a."userId" '
        'WHERE a."providerId" = %s AND a."accountId" = %s LIMIT 1',
        (PROVIDER, sub), fetch_all=True,
    )
    return filas[0] if filas else None


def _por_correo(email: str) -> Optional[dict]:
    filas = execute_sql_query(
        'SELECT id::text AS id, email, name, banned FROM neon_auth."user" WHERE lower(email) = %s LIMIT 1',
        (email,), fetch_all=True,
    )
    return filas[0] if filas else None


def resolve_apple_user(identidad: dict, name: Optional[str] = None) -> dict:
    """`identidad` es la salida de `apple_auth.verify_apple_identity_token` (ya verificada).
    Devuelve `{user_id, email, name, created, linked}` o lanza `AppleIdentityError`.
    tooltip-anchor: P1-PLAN-LOTE-146-RESOLVE"""
    sub = str(identidad.get("sub") or "").strip()
    email = str(identidad.get("email") or "").strip().lower()
    if not sub:
        raise AppleIdentityError("apple_token_invalid")

    fila = _por_sub(sub)
    if fila:
        if _vetado(fila):
            raise AppleIdentityError("account_banned")
        return {"user_id": fila["id"], "email": fila.get("email"), "name": fila.get("name"), "created": False, "linked": False}

    # Desde aquí hace falta un correo, y que Apple lo dé por verificado: es la ÚNICA prueba de que
    # quien entra es dueño de la cuenta a la que se le va a enlazar.
    if not email:
        raise AppleIdentityError("apple_no_email")
    if not identidad.get("email_verified"):
        raise AppleIdentityError("apple_email_unverified")

    fila = _por_correo(email)
    if fila:
        if _vetado(fila):
            raise AppleIdentityError("account_banned")
        _enlazar(fila["id"], sub)
        logger.info(f"🍎 [P1-PLAN-LOTE-146] Apple enlazado a una cuenta existente (uid={fila['id'][:8]}…).")
        return {"user_id": fila["id"], "email": fila.get("email"), "name": fila.get("name"), "created": False, "linked": True}

    nombre = _nombre(name, email, bool(identidad.get("is_private_email")))
    try:
        nuevas = execute_sql_write(
            'INSERT INTO neon_auth."user" (name, email, "emailVerified") VALUES (%s, %s, TRUE) RETURNING id::text AS id',
            (nombre, email), returning=True,
        )
    except Exception as e:
        # Dos peticiones a la vez (doble toque): la otra ya creó la fila y el UNIQUE del correo saltó.
        logger.info(f"[P1-PLAN-LOTE-146] alta de identidad chocó ({type(e).__name__}); se reintenta por correo.")
        nuevas = None
    if not nuevas:
        fila = _por_correo(email)
        if not fila or _vetado(fila):
            raise AppleIdentityError("apple_signup_failed")
        if not _por_sub(sub):
            _enlazar(fila["id"], sub)
        return {"user_id": fila["id"], "email": fila.get("email"), "name": fila.get("name"), "created": False, "linked": True}
    uid = nuevas[0]["id"]
    # Si este INSERT fallara, la identidad queda sin enlace: el próximo intento la encuentra POR CORREO
    # (camino 2) y la enlaza — se cura solo, no deja a nadie fuera.
    _enlazar(uid, sub)
    logger.info(f"🍎 [P1-PLAN-LOTE-146] identidad nueva por Apple (uid={uid[:8]}…, relay={bool(identidad.get('is_private_email'))}).")
    return {"user_id": uid, "email": email, "name": nombre, "created": True, "linked": True}
