# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-843 · 2026-09-29] El permiso para enviar datos personales a la IA de terceros — SSOT.

POR QUÉ
    Apple (App Review 5.1.2(i), texto de nov-2025) exige decir a qué IA de terceros van los datos personales y obtener
    permiso explícito ANTES del primer envío; el revisor lo prueba sobre una instalación limpia. El RGPD pide lo mismo
    por dos vías: consentimiento explícito para los datos de salud (art. 9(2)(a)) y, para la transferencia al
    proveedor de IA en China (sin decisión de adecuación ni cláusulas firmadas), consentimiento informado de sus
    riesgos (art. 49(1)(a)).
    Auditoría 2026-09-29, fila 4 y §A.1. Doc: `backend/docs/consentimiento_ia.md`.

QUÉ ES «VIGENTE»
    La cuenta aceptó la versión ACTUAL (`AI_CONSENT_VERSION`) con las DOS claves de IA (`ai_processing` y
    `ai_transfer_cn`) y no lo retiró después. `analytics` va aparte, es opcional y no cuenta para la IA.

DÓNDE VIVE
    `public.user_consents` es el registro de solo inserción (cada decisión, una fila: demostrable, art. 7.1) y las
    columnas `user_profiles.ai_consent_*` / `analytics_consent` son el estado derivado, para que decidir cueste UNA
    lectura por clave primaria. Las dos se escriben en la MISMA transacción y solo desde aquí.

CÓMO SE HACE CUMPLIR (no basta con la interfaz)
    - Peticiones: `requiere_consentimiento_ia` (428 `ai_consent_required`) en cada endpoint cuyo propósito es llamar a
      un proveedor; `hay_permiso_ia` (sin error) donde la IA es un efecto lateral que se puede saltar. Invitados: la
      cabecera `X-Bioboros-AI-Consent: <versión>`; el registro de su permiso lo guarda `registrar_invitado`.
    - Segundo plano: `condicion_sql_permiso` en la recogida del chunk worker (y en lo que avisa de esa cola) y
      `permite_ia(user_id, donde)` antes de cada llamada al proveedor de los crons y de los hilos de fondo.
    - Embeddings desde caminos SIN IA (lista de compras, Nevera, diario): `shopping_calculator.normalize_name` manda el
      nombre de un alimento a Cohere en su intento 6. Esos caminos llevan la MARCA del usuario
      (`embeddings_de_usuario`, `por_usuario` en los bucles de los crons, `embeddings_de_la_peticion` en los
      endpoints) y el intento 6 se salta sin su permiso. Sin marca = permitido: el pipeline y el worker ya van
      filtrados antes (428 / SQL).
    - Knob `MEALFIT_AI_CONSENT_GATE` (`off|log|block`, default `block`), ver `modo()`.

RETIRAR
    Bandera primero (`ai_consent_revoked_at`) y, en la MISMA transacción, el generador a 'tracking' con la hora de esa
    pausa en `ai_consent_paused_at`; después `plan_mode.pause_plan_generation` (cancela la cola con su firma, suelta
    los locks y sella el plan). No borra datos: eso sigue siendo «Eliminar cuenta». Volver a conceder reanuda SOLO si
    el plan sigue en 'tracking' desde esa hora exacta (un encendido o apagado a mano después la deja atrás).
"""
from __future__ import annotations

import hashlib
import logging
import re
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Optional

from fastapi import Depends, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from auth import get_verified_user_id
from db import execute_sql_query, execute_sql_write
from knobs import _env_str

logger = logging.getLogger(__name__)

#: La versión del texto aceptado. Subirla vuelve a pedir el permiso a TODOS y el backend rechaza la vieja: cambiar de
#: proveedor (también por knob), de datos o de país es cambiar de versión, y se sube ANTES de activar el cambio.
#: Espejo en `frontend/src/consent/version.js` (test de paridad).
AI_CONSENT_VERSION = "ia-2026-10-voz"

CLAVES = ("ai_processing", "ai_transfer_cn", "analytics")
CLAVES_IA = ("ai_processing", "ai_transfer_cn")
PLATAFORMAS = ("ios", "android", "web")
MODOS = ("off", "log", "block")
CABECERA_INVITADO = "X-Bioboros-AI-Consent"
ERROR_REQUERIDO = "ai_consent_required"

#: `detail` legible junto al código: un frontend viejo que no conoce el 428 enseña esta frase en vez de romperse.
MENSAJE_REQUERIDO = (
    "Para usar la IA necesitamos tu permiso para enviar tus datos a los proveedores de IA. "
    "Si no ves la pantalla para darlo, actualiza la app."
)
MENSAJE_NO_DISPONIBLE = "No pudimos comprobar tu permiso para usar la IA. Inténtalo de nuevo en unos segundos."

#: De dónde sale cada fila del registro (`user_consents.origen`): la decisión de la cuenta, la del invitado (con su
#: `guest_hash`) o la del invitado copiada a su cuenta al adoptar el plan, con su fecha original.
ORIGENES = ("cuenta", "invitado", "adopcion")

_RE_VERSION = re.compile(r"^[a-z0-9][a-z0-9.-]{0,31}$")
_RE_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_RE_LOCALE = re.compile(r"^[A-Za-z]{2,3}(-[A-Za-z0-9]{2,8})?$")
_RE_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
_RE_SESION_INVITADO = re.compile(r"^[A-Za-z0-9_-]{8,128}$")
_RE_COLUMNA = re.compile(r"^[a-z_][a-z0-9_]*(\.[a-z_][a-z0-9_]*)?$")
_MAX_APP_BUILD = 64
_PASAN = frozenset({"vigente", "invitado_con_cabecera"})

_COLUMNAS = "ai_consent_version, ai_consent_at, ai_consent_revoked_at, ai_cn_transfer_at, analytics_consent"
_INSERTAR_EN = (
    "INSERT INTO public.user_consents (user_id, guest_hash, consent_key, version, granted, text_sha256, locale, "
    "platform, app_build, origen) VALUES "
)
_FILA = "(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
_INSERTAR = _INSERTAR_EN + _FILA


def modo() -> str:
    """`MEALFIT_AI_CONSENT_GATE`. `block` (default): sin permiso vigente no sale nada hacia la IA. `log`: la falta de
    permiso solo se anota —es el modo del despliegue gradual—, pero una retirada EXPLÍCITA se respeta igual: es la
    decisión de la persona, no un permiso que aún no se le pidió. `off`: no hace nada (interruptor de emergencia)."""
    return _env_str("MEALFIT_AI_CONSENT_GATE", "block", choices=set(MODOS))


class ErrorDeConsentimiento(HTTPException):
    """Error del permiso con cuerpo PLANO `{error_code, version, detail}` (lo pinta el manejador de `instalar`)."""

    def __init__(self, status_code: int, error_code: str, mensaje: str):
        self.error_code = error_code
        self.mensaje = mensaje
        super().__init__(status_code=status_code, detail=cuerpo_del_error(error_code, mensaje))


class PerfilInexistente(Exception):
    """La cuenta no tiene fila en `user_profiles`."""


def cuerpo_del_error(error_code: str, mensaje: str) -> dict:
    return {"error_code": error_code, "version": AI_CONSENT_VERSION, "detail": mensaje}


async def _responder(request: Request, exc: ErrorDeConsentimiento) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content=cuerpo_del_error(exc.error_code, exc.mensaje))


def instalar(app) -> None:
    """Registra el manejador que da al 428 (y a los demás errores del permiso) su cuerpo plano."""
    app.add_exception_handler(ErrorDeConsentimiento, _responder)


# ───────────────────────────────────────────────────────────────── lectura del estado
def _uid(valor) -> Optional[str]:
    s = str(valor).strip().lower() if valor is not None else ""
    return s if _RE_UUID.match(s) else None


def _leer_fila(uid: str) -> Optional[dict]:
    """El estado derivado de la cuenta. LANZA si la base falla: quien llama decide (503 en una petición, «no» en un
    cron). Una lectura por clave primaria: es lo que lo hace barato en cada petición y en cada tick."""
    return execute_sql_query(f"SELECT {_COLUMNAS} FROM user_profiles WHERE id = %s", (uid,), fetch_one=True)


def _fila_vigente(fila) -> bool:
    fila = fila or {}
    return (fila.get("ai_consent_version") == AI_CONSENT_VERSION
            and fila.get("ai_consent_at") is not None
            and fila.get("ai_cn_transfer_at") is not None
            and fila.get("ai_consent_revoked_at") is None)


def _motivo(fila) -> str:
    if _fila_vigente(fila):
        return "vigente"
    fila = fila or {}
    if fila.get("ai_consent_revoked_at") is not None:
        return "retirado"
    return "version_vieja" if fila.get("ai_consent_version") else "sin_permiso"


def _bloquea(m: str, motivo: str) -> bool:
    if motivo in _PASAN or m == "off":
        return False
    if m == "log":
        return motivo == "retirado"
    return True


def _iso(valor) -> Optional[str]:
    if valor is None:
        return None
    if isinstance(valor, datetime):
        return (valor.astimezone(timezone.utc) if valor.tzinfo else valor.replace(tzinfo=timezone.utc)).isoformat()
    return str(valor)


def estado_de_fila(fila) -> dict:
    """La forma pública del estado (`GET /api/consents` y `profile.ai_consent`). Pura: vale para la fila propia y para
    el perfil ya leído por `get_user_profile` (fechas en texto ISO)."""
    fila = fila or {}
    analytics = fila.get("analytics_consent")
    return {
        "version": AI_CONSENT_VERSION,
        "vigente": _fila_vigente(fila),
        "ai_consent_version": fila.get("ai_consent_version"),
        "ai_consent_at": _iso(fila.get("ai_consent_at")),
        "ai_cn_transfer_at": _iso(fila.get("ai_cn_transfer_at")),
        "ai_consent_revoked_at": _iso(fila.get("ai_consent_revoked_at")),
        "analytics": analytics if isinstance(analytics, bool) else None,
    }


def estado(user_id) -> dict:
    """El estado de la cuenta. LANZA si la base falla."""
    uid = _uid(user_id)
    return estado_de_fila(_leer_fila(uid) if uid else None)


def vigente(user_id) -> bool:
    """¿La cuenta puede usar la IA? Versión actual, las dos claves y sin retirar. Ilegible ⇒ False (fail-closed)."""
    uid = _uid(user_id)
    if uid is None:
        return False
    try:
        return _fila_vigente(_leer_fila(uid))
    except Exception as e:  # noqa: BLE001
        logger.warning(f"⚠️ [P1-PLAN-LOTE-843] permiso de IA ilegible para {uid[:8]} ({e!r}): cuenta como NO vigente")
        return False


def permite_ia(user_id, donde: str = "") -> bool:
    """El gate de SEGUNDO PLANO: ¿puede este trabajo llamar al proveedor para esta cuenta? Se pregunta justo antes de
    la llamada. `off` ⇒ sí; `log` ⇒ sí salvo retirada explícita; `block` ⇒ solo con permiso vigente. Fail-closed en
    `block` (base ilegible o id que no es de una cuenta ⇒ no). Nunca lanza."""
    m = modo()
    if m == "off":
        return True
    uid = _uid(user_id)
    if uid is None:
        return m != "block"
    try:
        motivo = _motivo(_leer_fila(uid))
    except Exception as e:  # noqa: BLE001
        logger.warning(f"⚠️ [P1-PLAN-LOTE-843] {donde}: permiso de IA ilegible para {uid[:8]} ({e!r})")
        return m != "block"
    if _bloquea(m, motivo):
        logger.debug(f"[P1-PLAN-LOTE-843] {donde}: {uid[:8]} sin permiso de IA ({motivo}); no se llama al proveedor")
        return False
    if motivo not in _PASAN:
        logger.debug(f"[P1-PLAN-LOTE-843] {donde}: {uid[:8]} sin permiso de IA ({motivo}); modo {m}, pasa")
    return True


# ───────────────────────────────────────────────────────────────── embeddings desde caminos sin IA
class _MarcaEmbeddings:
    """De quién es el trabajo en curso. La decisión se toma la PRIMERA vez que alguien la pide (el intento 6 de
    `normalize_name` es raro: la mayoría de las peticiones nunca lo alcanzan) y se recuerda para el resto: una lectura
    por clave primaria como mucho, por petición o por vuelta del bucle."""

    __slots__ = ("user_id", "cabecera", "donde", "_permite")

    def __init__(self, user_id, cabecera=None, donde: str = ""):
        self.user_id, self.cabecera, self.donde, self._permite = user_id, cabecera, donde, None

    def permite(self) -> bool:
        if self._permite is None:
            self._permite = _permite_mandar_texto(self.user_id, self.cabecera, self.donde)
        return self._permite


#: La marca del usuario del trabajo en curso; None = sin marca (el pipeline y el worker, ya filtrados por el 428 y la
#: recogida): permitido. La leen los que mandan texto a un proveedor desde caminos que NO son de IA.
_EMBEDDINGS_DE: ContextVar = ContextVar("p1_plan_lote_843_embeddings_de", default=None)


def _permite_mandar_texto(user_id, cabecera, donde) -> bool:
    """La misma decisión que el 428 y `hay_permiso_ia` (cuenta ⇒ la base; invitado ⇒ la cabecera), sin error ni ruido.
    Nunca lanza: base ilegible ⇒ no en `block`, sí en `log`."""
    m = modo()
    if m == "off":
        return True
    try:
        motivo = _decidir_peticion(user_id, cabecera)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"⚠️ [P1-PLAN-LOTE-843] {donde}: permiso de IA ilegible para los embeddings ({e!r})")
        return m != "block"
    if _bloquea(m, motivo):
        logger.info(f"⏭️ [P1-PLAN-LOTE-843] {donde}: sin permiso de IA ({motivo}); los nombres no van a Cohere")
        return False
    return True


@contextmanager
def embeddings_de_usuario(user_id, donde: str = ""):
    """Marca el trabajo de `user_id` mientras dura el bloque: `embeddings_permitidos()` responde por ÉL. Es lo que
    envuelve cada punto de entrada que llega a los embeddings sin pasar por un endpoint de IA (en los crons, por
    usuario dentro del bucle: `por_usuario`)."""
    token = _EMBEDDINGS_DE.set(_MarcaEmbeddings(user_id, None, donde))
    try:
        yield
    finally:
        try:
            _EMBEDDINGS_DE.reset(token)
        except ValueError:  # cerrado desde otro contexto (un generador recogido fuera): no hay nada que restaurar
            pass


def por_usuario(filas, clave: str = "user_id", donde: str = ""):
    """Recorre `filas` con la marca del `clave` de CADA fila puesta durante su vuelta del bucle. Es la forma de
    envolver por usuario el cuerpo de un bucle de cron sin reindentarlo: `for fila in por_usuario(filas):`."""
    for fila in filas or ():
        with embeddings_de_usuario(fila.get(clave) if isinstance(fila, dict) else None, donde):
            yield fila


def embeddings_permitidos() -> bool:
    """¿Puede el trabajo en curso mandar texto a un proveedor de embeddings? Sin marca, sí (ver `_EMBEDDINGS_DE`)."""
    marca = _EMBEDDINGS_DE.get()
    return True if marca is None else marca.permite()


# ───────────────────────────────────────────────────────────────── SQL de los gates de la cola
def condicion_sql_permiso(columna_usuario: str) -> str:
    """Expresión SQL «la cuenta de `columna_usuario` puede usar la IA», según el modo; '' con `off`. CONSTANTE: la
    columna se valida como identificador y la versión es la de este módulo (sin comillas posibles): cero entrada de
    usuario en el SQL."""
    if not _RE_COLUMNA.match(columna_usuario or ""):
        raise ValueError(f"columna no válida para el gate del permiso: {columna_usuario!r}")
    if not _RE_VERSION.match(AI_CONSENT_VERSION):
        raise ValueError("AI_CONSENT_VERSION no puede ir como literal en SQL")
    m = modo()
    if m == "off":
        return ""
    if m == "log":
        return (f"NOT EXISTS (SELECT 1 FROM public.user_profiles upc WHERE upc.id = {columna_usuario} "
                "AND upc.ai_consent_revoked_at IS NOT NULL)")
    return (f"EXISTS (SELECT 1 FROM public.user_profiles upc WHERE upc.id = {columna_usuario} "
            f"AND upc.ai_consent_version = '{AI_CONSENT_VERSION}' AND upc.ai_consent_at IS NOT NULL "
            "AND upc.ai_cn_transfer_at IS NOT NULL AND upc.ai_consent_revoked_at IS NULL)")


def fragmento_sql_permiso(columna_usuario: str) -> str:
    """`AND <condición>` en su propia línea, listo para un WHERE; '' con `off`."""
    c = condicion_sql_permiso(columna_usuario)
    return f"\n                -- [P1-PLAN-LOTE-843] permiso de IA de terceros\n                AND {c}" if c else ""


# ───────────────────────────────────────────────────────────────── escribir decisiones
def validar_peticion(data) -> dict:
    """Valida el cuerpo de `POST /api/consents` y `POST /api/consents/guest`. Conceder la IA exige las DOS claves a
    true; `analytics` es opcional e independiente. Lanza `ErrorDeConsentimiento` (409 versión vieja, 422 el resto)."""
    def _mal(codigo, mensaje):
        return ErrorDeConsentimiento(422, codigo, mensaje)

    if not isinstance(data, dict):
        raise _mal("ai_consent_invalid_field", "El cuerpo tiene que ser un objeto JSON.")
    if data.get("version") != AI_CONSENT_VERSION:
        raise ErrorDeConsentimiento(409, "ai_consent_version_outdated",
                                    "Ese texto del permiso ya no es el vigente: actualiza la app para leer el nuevo.")
    for clave in CLAVES:
        v = data.get(clave)
        if v is not None and not isinstance(v, bool):
            raise _mal("ai_consent_invalid_field", f"«{clave}» tiene que ser true o false.")
    p, cn = data.get("ai_processing"), data.get("ai_transfer_cn")
    if p is None and cn is None:
        ia = False
    elif p is True and cn is True:
        ia = True
    else:
        raise _mal("ai_consent_incomplete", "Para activar la IA hacen falta las dos casillas. "
                                            "Para quitar el permiso, usa «Retirar mi permiso».")
    analytics = data.get("analytics")
    if not ia and analytics is None:
        raise _mal("ai_consent_nothing_to_record", "No hay ninguna decisión que registrar.")
    sha = data.get("text_sha256")
    if sha is not None:
        sha = sha.strip().lower() if isinstance(sha, str) else sha
        if not isinstance(sha, str) or not _RE_HEX64.match(sha):
            raise _mal("ai_consent_invalid_field", "«text_sha256» tiene que ser un SHA-256 en hexadecimal.")
    return {"ia": ia, "analytics": analytics, "text_sha256": sha, **validar_contexto(data)}


def validar_contexto(data) -> dict:
    """`locale`, `platform` y `app_build` (opcionales): desde dónde se tomó la decisión. 422 si no tienen forma."""
    data = data if isinstance(data, dict) else {}
    locale = data.get("locale")
    if locale is not None and not (isinstance(locale, str) and len(locale) <= 16 and _RE_LOCALE.match(locale)):
        raise ErrorDeConsentimiento(422, "ai_consent_invalid_field", "«locale» no es un idioma válido.")
    platform = data.get("platform")
    if platform is not None and platform not in PLATAFORMAS:
        raise ErrorDeConsentimiento(422, "ai_consent_invalid_field", "«platform» tiene que ser ios, android o web.")
    app_build = data.get("app_build")
    if isinstance(app_build, int) and not isinstance(app_build, bool):
        app_build = str(app_build)
    if app_build is not None:
        if not isinstance(app_build, str) or not app_build.strip() or len(app_build.strip()) > _MAX_APP_BUILD:
            raise ErrorDeConsentimiento(422, "ai_consent_invalid_field", "«app_build» no es válido.")
        app_build = app_build.strip()
    return {"locale": locale, "platform": platform, "app_build": app_build}


def _en_transaccion(fn):
    from db_core import connection_pool  # como plan_mode: el pool se lee al usarlo, no al importar
    from psycopg.rows import dict_row
    if not connection_pool:
        raise RuntimeError("sin pool de base de datos")
    with connection_pool.connection() as conn:
        with conn.transaction():
            with conn.cursor(row_factory=dict_row) as cur:
                return fn(cur)


def _decisiones(ia: bool, analytics) -> list:
    return ([(c, True) for c in CLAVES_IA] if ia else []) + (
        [("analytics", bool(analytics))] if analytics is not None else [])


def _pausa_de_la_retirada(fila) -> bool:
    """¿La pausa vigente del generador la puso la retirada del permiso? Solo si el plan SIGUE en 'tracking' desde esa
    retirada: `plan_mode_changed_at` es exactamente la hora que la retirada estampó en `ai_consent_paused_at` (las dos
    con el mismo `now()` de su transacción). Un encendido o un apagado a mano después cambian `plan_mode_changed_at`;
    quien ya estaba en seguimiento al retirar no tiene marca. Determinista: sin ventanas de tiempo."""
    fila = fila or {}
    pausa = fila.get("ai_consent_paused_at")
    return fila.get("plan_mode") == "tracking" and pausa is not None and fila.get("plan_mode_changed_at") == pausa


def _interruptor_del_plan() -> bool:
    """`MEALFIT_PLAN_MODE_SWITCH`: con el interruptor apagado el modo del plan no existe y la retirada no lo toca."""
    try:
        from plan_mode import PLAN_MODE_SWITCH_ENABLED
        return bool(PLAN_MODE_SWITCH_ENABLED)
    except Exception:  # noqa: BLE001
        return False


def registrar(user_id, *, ia: bool, analytics: Optional[bool] = None, locale: Optional[str] = None,
              platform: Optional[str] = None, app_build: Optional[str] = None,
              text_sha256: Optional[str] = None) -> dict:
    """Anota la decisión de la cuenta (ya validada por `validar_peticion`) y deriva su estado, en UNA transacción. Si
    la retirada había pausado el generador y sigue en esa pausa, conceder lo reanuda; conceder la IA limpia siempre la
    marca de la pausa. Devuelve el estado + `plan_reanudado` + `plan_expired` (el de `resume_plan_generation`: la pausa
    duró más que la ventana de reanudación y el plan hay que renovarlo)."""
    uid = _uid(user_id)
    if uid is None:
        raise ValueError("user_id no es un UUID")
    decisiones = _decisiones(ia, analytics)
    if not decisiones:
        raise ValueError("nada que registrar")

    def _tx(cur):
        cur.execute("SELECT plan_mode, plan_mode_changed_at, ai_consent_paused_at FROM user_profiles "
                    "WHERE id = %s FOR UPDATE", (uid,))
        antes = cur.fetchone()
        if not antes:
            raise PerfilInexistente(uid)
        for clave, concedido in decisiones:
            cur.execute(_INSERTAR, (uid, None, clave, AI_CONSENT_VERSION, concedido, text_sha256, locale, platform,
                                    app_build, "cuenta"))
        sets, params = [], []
        if ia:
            sets += ["ai_consent_version = %s", "ai_consent_at = now()", "ai_cn_transfer_at = now()",
                     "ai_consent_revoked_at = NULL", "ai_consent_paused_at = NULL"]
            params.append(AI_CONSENT_VERSION)
        if analytics is not None:
            sets.append("analytics_consent = %s")
            params.append(bool(analytics))
        cur.execute(f"UPDATE user_profiles SET {', '.join(sets)} WHERE id = %s RETURNING {_COLUMNAS}",
                    (*params, uid))
        return antes, cur.fetchone()

    antes, despues = _en_transaccion(_tx)
    reanudado = expirado = False
    if ia and _pausa_de_la_retirada(antes):
        try:
            from plan_mode import resume_plan_generation
            r = resume_plan_generation(uid) or {}
            reanudado = r.get("plan_mode") == "plan" and not r.get("skipped")
            expirado = bool(reanudado and r.get("plan_expired"))
        except Exception as e:  # noqa: BLE001
            logger.error(f"❌ [P1-PLAN-LOTE-843] {uid[:8]}: permiso concedido pero el plan NO se reanudó ({e!r}); "
                         f"queda en pausa y se reanuda a mano desde Configuración.")
    logger.info(f"✅ [P1-PLAN-LOTE-843] {uid[:8]}: permiso anotado (ia={ia}, analytics={analytics}, "
                f"plan_reanudado={reanudado}, plan_expired={expirado})")
    out = estado_de_fila(despues)
    out["plan_reanudado"] = reanudado
    out["plan_expired"] = expirado
    return out


def retirar(user_id, *, locale: Optional[str] = None, platform: Optional[str] = None,
            app_build: Optional[str] = None) -> dict:
    """Retira el permiso de IA. En UNA transacción: la bandera (con las filas del registro) y, si el generador estaba
    encendido, su paso a 'tracking' con la hora de esa pausa en `ai_consent_paused_at` (el mismo `now()` que
    `plan_mode_changed_at`). Desde ese COMMIT la recogida de bloques y los crons ya dicen que no. Después, la cola, los
    locks y el sello con `pause_plan_generation` (su UPDATE de la bandera ya no cambia nada). Retirar dos veces conserva
    la fecha de la primera y no escribe filas nuevas. No borra datos. Devuelve el estado + `plan_pausado`: true solo si
    ESTA retirada apagó el generador (quien ya estaba en seguimiento no tenía nada que pausar)."""
    uid = _uid(user_id)
    if uid is None:
        raise ValueError("user_id no es un UUID")
    interruptor = _interruptor_del_plan()

    def _tx(cur):
        cur.execute("SELECT ai_consent_revoked_at, plan_mode FROM user_profiles WHERE id = %s FOR UPDATE", (uid,))
        antes = cur.fetchone()
        if not antes:
            raise PerfilInexistente(uid)
        if antes.get("ai_consent_revoked_at") is None:
            cur.execute("UPDATE user_profiles SET ai_consent_revoked_at = now() WHERE id = %s", (uid,))
            for clave in CLAVES_IA:
                cur.execute(_INSERTAR, (uid, None, clave, AI_CONSENT_VERSION, False, None, locale, platform,
                                        app_build, "cuenta"))
        apaga = interruptor and antes.get("plan_mode") != "tracking"
        if apaga:
            cur.execute("UPDATE user_profiles SET plan_mode = 'tracking', plan_mode_changed_at = now(), "
                        "ai_consent_paused_at = now() WHERE id = %s", (uid,))
        cur.execute(f"SELECT {_COLUMNAS} FROM user_profiles WHERE id = %s", (uid,))
        return cur.fetchone(), apaga

    fila, plan_pausado = _en_transaccion(_tx)
    if interruptor:
        try:
            from plan_mode import pause_plan_generation
            pause_plan_generation(uid)
        except Exception as e:  # noqa: BLE001
            logger.error(f"❌ [P1-PLAN-LOTE-843] {uid[:8]}: permiso retirado pero la cola NO se canceló ({e!r}); "
                         f"el gate de la recogida y el de plan_mode la frenan igual.")
    logger.info(f"⏸ [P1-PLAN-LOTE-843] {uid[:8]}: permiso de IA retirado (plan_pausado={plan_pausado})")
    out = estado_de_fila(fila)
    out["plan_pausado"] = bool(plan_pausado)
    return out


def hash_de_sesion(session_id) -> Optional[str]:
    """sha256 del session_id del invitado: lo que se guarda en `guest_hash`. El id crudo NUNCA se guarda."""
    s = session_id.strip() if isinstance(session_id, str) else ""
    if not _RE_SESION_INVITADO.match(s):
        return None
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def registrar_invitado(session_id, *, ia: bool, analytics: Optional[bool] = None, locale: Optional[str] = None,
                       platform: Optional[str] = None, app_build: Optional[str] = None,
                       text_sha256: Optional[str] = None) -> dict:
    """Anota el permiso del INVITADO con `guest_hash`. En sus llamadas a la IA viaja la cabecera; esta fila es la
    prueba, y la que `adoptar_de_invitado` pasa a la cuenta con su fecha. Una sola sentencia: atómica."""
    h = hash_de_sesion(session_id)
    if h is None:
        raise ValueError("session_id de invitado no válido")
    decisiones = _decisiones(ia, analytics)
    if not decisiones:
        raise ValueError("nada que registrar")
    params: list = []
    for clave, concedido in decisiones:
        params += [None, h, clave, AI_CONSENT_VERSION, concedido, text_sha256, locale, platform, app_build, "invitado"]
    execute_sql_write(_INSERTAR_EN + ", ".join([_FILA] * len(decisiones)), tuple(params))
    logger.info(f"✅ [P1-PLAN-LOTE-843] invitado {h[:8]}: permiso anotado (ia={ia}, analytics={analytics})")
    return {"ok": True, "version": AI_CONSENT_VERSION, "header": CABECERA_INVITADO, "ai": ia, "analytics": analytics}


def adoptar_de_invitado(session_id, user_id) -> dict:
    """Al adoptar el plan del invitado en su cuenta, su permiso pasa con la FECHA original (`created_at` copiado),
    marcado `origen = 'adopcion'` y sin duplicarse si se llama dos veces. El estado de la cuenta se toma del invitado
    solo si su permiso de IA es de la versión actual y la cuenta no tiene una decisión propia MÁS RECIENTE; la
    analítica, solo si nunca se preguntó. Nunca lanza: adoptar el plan no puede caerse por esto."""
    vacio = {"adoptadas": 0, "estado_actualizado": False}
    uid, h = _uid(user_id), hash_de_sesion(session_id)
    if uid is None or h is None:
        return vacio

    def _tx(cur):
        cur.execute("SELECT consent_key, version, granted, text_sha256, locale, platform, app_build, created_at "
                    "FROM public.user_consents WHERE guest_hash = %s ORDER BY created_at ASC", (h,))
        filas = cur.fetchall() or []
        n = 0
        for f in filas:
            cur.execute(
                "INSERT INTO public.user_consents (user_id, consent_key, version, granted, text_sha256, locale, "
                "platform, app_build, created_at, origen) SELECT %s::uuid, %s, %s, %s, %s, %s, %s, %s, "
                "%s::timestamptz, 'adopcion' "
                "WHERE NOT EXISTS (SELECT 1 FROM public.user_consents WHERE user_id = %s::uuid AND consent_key = %s "
                "AND version = %s AND created_at = %s::timestamptz)",
                (uid, f["consent_key"], f["version"], f["granted"], f.get("text_sha256"), f.get("locale"),
                 f.get("platform"), f.get("app_build"), f["created_at"],
                 uid, f["consent_key"], f["version"], f["created_at"]))
            n += max(0, int(cur.rowcount or 0))
        ultima = {f["consent_key"]: f for f in filas}  # en orden ASC: gana la más reciente de cada clave
        p, c = ultima.get("ai_processing"), ultima.get("ai_transfer_cn")
        actualizado = False
        if (p and c and p["granted"] and c["granted"]
                and p["version"] == AI_CONSENT_VERSION and c["version"] == AI_CONSENT_VERSION):
            fecha = max(p["created_at"], c["created_at"])
            cur.execute(
                "UPDATE user_profiles SET ai_consent_version = %s, ai_consent_at = %s, ai_cn_transfer_at = %s, "
                "ai_consent_revoked_at = NULL WHERE id = %s AND (ai_consent_at IS NULL OR ai_consent_at < %s) "
                "AND (ai_consent_revoked_at IS NULL OR ai_consent_revoked_at < %s)",
                (AI_CONSENT_VERSION, p["created_at"], c["created_at"], uid, fecha, fecha))
            actualizado = int(cur.rowcount or 0) > 0
        a = ultima.get("analytics")
        if a is not None:
            cur.execute("UPDATE user_profiles SET analytics_consent = %s WHERE id = %s AND analytics_consent IS NULL",
                        (bool(a["granted"]), uid))
        return n, actualizado

    try:
        n, actualizado = _en_transaccion(_tx)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"⚠️ [P1-PLAN-LOTE-843] permiso del invitado no adoptado por {uid[:8]} ({e!r})")
        return {**vacio, "error": True}
    if n or actualizado:
        logger.info(f"✅ [P1-PLAN-LOTE-843] {uid[:8]}: {n} decisión(es) del invitado adoptadas "
                    f"(estado_actualizado={actualizado})")
    return {"adoptadas": n, "estado_actualizado": actualizado}


# ───────────────────────────────────────────────────────────────── dependencias de FastAPI
def _decidir_peticion(verified_user_id, cabecera) -> str:
    """El motivo de la petición. Con cuenta manda la base (la cabecera se ignora); sin cuenta, la cabecera. LANZA si
    la base falla."""
    uid = _uid(verified_user_id)
    if uid:
        return _motivo(_leer_fila(uid))
    c = cabecera.strip() if isinstance(cabecera, str) else ""
    if c == AI_CONSENT_VERSION:
        return "invitado_con_cabecera"
    return "invitado_version_vieja" if c else "invitado_sin_cabecera"


def _ruta(request) -> str:
    try:
        return request.url.path
    except Exception:  # noqa: BLE001
        return "?"


def requiere_consentimiento_ia(
    request: Request,
    verified_user_id: Optional[str] = Depends(get_verified_user_id),
    x_bioboros_ai_consent: Optional[str] = Header(None, alias=CABECERA_INVITADO),
) -> None:
    """Dependencia de los endpoints que llaman a un proveedor de IA. Sin permiso vigente ⇒ 428
    `{"error_code": "ai_consent_required", "version": AI_CONSENT_VERSION, "detail": ...}`. Invitado: la cabecera
    `X-Bioboros-AI-Consent` con la versión actual. `log` solo lo anota (salvo retirada explícita); `off` nada."""
    m = modo()
    if m == "off":
        return None
    try:
        motivo = _decidir_peticion(verified_user_id, x_bioboros_ai_consent)
    except Exception as e:  # noqa: BLE001
        logger.error(f"❌ [P1-PLAN-LOTE-843] {_ruta(request)}: no se pudo leer el permiso de IA ({e!r})")
        if m == "block":
            raise ErrorDeConsentimiento(503, "ai_consent_unavailable", MENSAJE_NO_DISPONIBLE)
        return None
    if not _bloquea(m, motivo):
        if motivo not in _PASAN:  # `info`, no `warning`: en `log` saldría en cada petición durante el despliegue
            logger.info(f"📝 [P1-PLAN-LOTE-843] {_ruta(request)}: sin permiso de IA ({motivo}); modo log, pasa")
        return None
    logger.info(f"⛔ [P1-PLAN-LOTE-843] {_ruta(request)}: 428 ({motivo})")
    raise ErrorDeConsentimiento(428, ERROR_REQUERIDO, MENSAJE_REQUERIDO)


def hay_permiso_ia(
    request: Request,
    verified_user_id: Optional[str] = Depends(get_verified_user_id),
    x_bioboros_ai_consent: Optional[str] = Header(None, alias=CABECERA_INVITADO),
) -> bool:
    """La misma decisión sin error, para los endpoints donde la IA es un EFECTO LATERAL que se puede saltar (traducir
    lo que se muestra): el endpoint responde igual, sin la parte de IA. Quien lo usa comprueba `is False`: llamado como
    función en un test, el valor por defecto es el `Depends` y la conducta es la de siempre."""
    m = modo()
    if m == "off":
        return True
    try:
        motivo = _decidir_peticion(verified_user_id, x_bioboros_ai_consent)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"⚠️ [P1-PLAN-LOTE-843] {_ruta(request)}: permiso de IA ilegible ({e!r})")
        return m != "block"
    if _bloquea(m, motivo):
        logger.info(f"⏭️ [P1-PLAN-LOTE-843] {_ruta(request)}: sin permiso de IA ({motivo}); se atiende sin la IA")
        return False
    if motivo not in _PASAN:  # `info`, no `warning` (ver `requiere_consentimiento_ia`)
        logger.info(f"📝 [P1-PLAN-LOTE-843] {_ruta(request)}: sin permiso de IA ({motivo}); modo log, pasa")
    return True


async def embeddings_de_la_peticion(
    request: Request,
    verified_user_id: Optional[str] = Depends(get_verified_user_id),
    x_bioboros_ai_consent: Optional[str] = Header(None, alias=CABECERA_INVITADO),
) -> None:
    """Dependencia de los endpoints SIN IA que llegan a los embeddings (lista de compras, Nevera, diario): marca la
    petición entera con su titular (cuenta ⇒ su permiso en la base; invitado ⇒ su cabecera). No lee nada al entrar: la
    decisión se toma la primera vez que el intento 6 de `normalize_name` la pide. `async` a propósito: la marca se pone
    en el contexto de la petición y FastAPI lo copia al hilo de un endpoint síncrono (una dependencia síncrona la
    pondría en un hilo aparte y se perdería). Sin reset: el contexto muere con la petición."""
    _EMBEDDINGS_DE.set(_MarcaEmbeddings(verified_user_id, x_bioboros_ai_consent, _ruta(request)))


__all__ = [
    "AI_CONSENT_VERSION", "CLAVES", "CLAVES_IA", "PLATAFORMAS", "MODOS", "ORIGENES", "CABECERA_INVITADO",
    "ERROR_REQUERIDO", "MENSAJE_REQUERIDO", "modo", "ErrorDeConsentimiento", "PerfilInexistente", "cuerpo_del_error",
    "instalar", "estado_de_fila", "estado", "vigente", "permite_ia", "condicion_sql_permiso", "fragmento_sql_permiso",
    "embeddings_de_usuario", "por_usuario", "embeddings_permitidos", "embeddings_de_la_peticion",
    "validar_peticion", "validar_contexto", "registrar", "retirar", "hash_de_sesion", "registrar_invitado",
    "adoptar_de_invitado",
    "requiere_consentimiento_ia", "hay_permiso_ia",
]
