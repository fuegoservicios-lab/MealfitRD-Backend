"""[P1-PLAN-LOTE-845 · 2026-09-29] Acceso de App Review: un código fijo para UNA cuenta de demostración.

Por qué (auditoría App Store 2026-09-29, fila 8.1, §A.2). Bioboros solo se entra con un código de un solo uso que
llega al correo (Neon Auth), con Google o con Apple. El revisor de Apple no tiene acceso al buzón de la cuenta de
prueba, así que no puede entrar, y eso es un rechazo seguro por la guideline 2.1. Aquí vive la excepción, acotada
a UN correo y a UN código que solo existen en el `.env` del VPS.

Contrato (el endpoint es `POST /api/auth/email-otp/verify`, `routers/auth_session.py`):
  - Inerte si falta cualquiera de `MEALFIT_REVIEW_LOGIN_EMAIL` o `MEALFIT_REVIEW_LOGIN_CODE_SHA256`, o si el hash no
    es un sha256 en hexadecimal. Inerte = el endpoint se comporta exactamente como antes.
  - El correo casa exacto salvo mayúsculas y espacios de los bordes. El código se compara con
    `hmac.compare_digest(sha256(código), hash)`, en tiempo constante.
  - Correo distinto o código distinto: el endpoint sigue su camino de siempre hacia Neon Auth, con la misma
    respuesta. Un código real de Neon para el correo de demostración sigue funcionando.
  - Con el código correcto se busca la cuenta EXISTENTE por correo en `neon_auth."user"`. Si no existe, o está
    vetada, no se crea: 401, igual que un código inválido (más la alerta `review_login_account_missing`).
  - La sesión la emite el endpoint con `set_session_cookie`, la misma que emiten el OTP y «Continuar con Apple».
  - Cada uso deja rastro: un `logger.warning` con el marcador y la alerta `review_login_used:<user_id>` en
    `system_alerts` (una fila por cuenta, `metadata.uses` cuenta los usos y cada uso la reabre).
  - El límite de ritmo es el del endpoint (`_OTP_VERIFY_LIMITER`, 10/60 s por IP).
  - [ronda 1] Además, los fallos con EL correo de demostración se cuentan por hora (Redis o memoria): a los 10,
    alerta `review_login_failed_burst`; a los 30 la rama del código fijo queda inerte hasta que la ventana se vacíe.
    Una cuenta de administración nunca recibe sesión por aquí (`review_login_account_privileged`).

SECRETOS, NO KNOBS. Las dos variables se leen con `os.environ`, a propósito, y no con `knobs._env_str`: el registro de
knobs se publica SIN autenticación en `/admin/knobs` (entero) y en `/health/version` (`knobs_diff`). El sha256 de un
código de 6 cifras se revierte probando el millón de combinaciones en menos de un segundo, así que publicar el hash
es publicar el código. `test_p1_plan_lote_845_revisor.py` comprueba que ninguna de las dos entra en el registro.

Cómo se calcula el hash (SIN salto de línea: `echo 123456 | sha256sum` mete uno y el hash no casaría nunca):
    python -c "import hashlib; print(hashlib.sha256(b'123456').hexdigest())"

El código tiene que ser de 6 cifras: el campo del login (`Login.jsx`) borra todo lo que no sea una cifra y envía solo
al llegar a 6. Con más cifras, el envío automático mandaría las 6 primeras y el revisor vería «Código inválido».
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import re
import threading
import time
from datetime import datetime, timezone
from typing import Optional

from db import execute_sql_query, execute_sql_write

logger = logging.getLogger(__name__)

ENV_CORREO = "MEALFIT_REVIEW_LOGIN_EMAIL"
ENV_HASH = "MEALFIT_REVIEW_LOGIN_CODE_SHA256"

_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_config_invalida_avisada = False


def configuracion() -> Optional[tuple[str, str]]:
    """`(correo, hash)` normalizados, o None si el acceso está apagado (falta una variable o no vale).

    Se lee en cada llamada: el coste es nulo y los tests pueden fijar el entorno con `monkeypatch.setenv`."""
    correo = (os.environ.get(ENV_CORREO) or "").strip().lower()
    esperado = (os.environ.get(ENV_HASH) or "").strip().lower()
    if not correo or not esperado:
        return None
    if "@" not in correo or not _HEX_SHA256.match(esperado):
        _avisar_config_invalida()
        return None
    return correo, esperado


def _avisar_config_invalida() -> None:
    """Una sola vez por proceso: la config mal puesta no debe llenar el log en cada intento de login."""
    global _config_invalida_avisada
    if _config_invalida_avisada:
        return
    _config_invalida_avisada = True
    logger.error(
        f"🛑 [P1-PLAN-LOTE-845] Acceso de App Review APAGADO: {ENV_CORREO} tiene que ser un correo y {ENV_HASH} "
        "un sha256 en hexadecimal (64 caracteres)."
    )


def coincide(email: str, codigo: str) -> bool:
    """True solo si el acceso está configurado, el correo es EL de la demostración, el código es el fijo y la rama no
    está bloqueada por fallos.

    El hash se calcula SIEMPRE, antes de mirar el correo, y se compara con `hmac.compare_digest` (tiempo constante),
    nunca con `==`. Hace I/O (Redis y, al cruzar un umbral, `system_alerts`) solo cuando el correo es el de la
    demostración: el que llama lo corre en un hilo. tooltip-anchor: P1-PLAN-LOTE-845-COMPARE"""
    digest = hashlib.sha256(str(codigo or "").encode("utf-8")).hexdigest()
    config = configuracion()
    if config is None:
        return False
    correo, esperado = config
    if str(email or "").strip().lower() != correo:
        return False
    acierto = hmac.compare_digest(digest, esperado)
    if fallos_en_ventana() >= UMBRAL_BLOQUEO:
        # Bloqueada: la rama del código fijo es inerte hasta que la ventana se vacíe. Ni el acierto vale ni el intento
        # cuenta (un intento que no se evalúa no prueba nada; contarlo solo prolongaría el bloqueo).
        _avisar_bloqueo()
        return False
    if not acierto:
        _anotar_fallo()
    return acierto


# ─────────────────────── [ronda 1] intentos fallidos contra el correo de demostración ───────────────────────
# Un código fijo de 6 cifras sin caducidad se adivina: el límite por IP del endpoint no para a quien rota IPs. Aquí se
# cuentan los fallos con EL correo de demostración en una ventana deslizante de una hora (Redis si hay, si no en
# memoria de este proceso): a los 10 salta `review_login_failed_burst`; a los 30 la rama del código fijo queda inerte
# hasta que la ventana se vacíe. El camino normal hacia Neon no cambia y la respuesta tampoco: el que llama sigue a Neon
# igual que con un código equivocado, así que no hay oráculo. Riesgo aceptado (ruling del controlador): quien conozca el
# correo puede dejar al revisor sin entrar mientras siga fallando; a cambio, adivinar el código pasa de horas a años.
VENTANA_S = 3600
UMBRAL_ALERTA = 10
UMBRAL_BLOQUEO = 30
_CLAVE_REDIS = "review_login:fallos"
_fallos_locales: list = []
_cerrojo = threading.Lock()
_bloqueo_avisado_hasta = 0.0


def _redis():
    try:
        from cache_manager import redis_client
        return redis_client
    except Exception:
        return None


def _local_purgar(ahora: float) -> None:
    _fallos_locales[:] = [t for t in _fallos_locales if ahora - t < VENTANA_S]


def fallos_en_ventana() -> int:
    """Fallos con el correo de demostración en la última hora."""
    ahora = time.time()
    r = _redis()
    if r is not None:
        try:
            r.zremrangebyscore(_CLAVE_REDIS, 0, ahora - VENTANA_S)
            return int(r.zcard(_CLAVE_REDIS))
        except Exception as e:
            logger.warning(f"⚠️ [P1-PLAN-LOTE-845] Redis falló contando fallos de revisión; memoria local: {e}")
    with _cerrojo:
        _local_purgar(ahora)
        return len(_fallos_locales)


def _anotar_fallo() -> None:
    ahora = time.time()
    total = None
    r = _redis()
    if r is not None:
        try:
            pipe = r.pipeline()
            pipe.zadd(_CLAVE_REDIS, {f"{ahora:.6f}:{os.getpid()}:{id(pipe)}": ahora})
            pipe.zremrangebyscore(_CLAVE_REDIS, 0, ahora - VENTANA_S)
            pipe.zcard(_CLAVE_REDIS)
            pipe.expire(_CLAVE_REDIS, VENTANA_S)
            total = int(pipe.execute()[2])
        except Exception as e:
            logger.warning(f"⚠️ [P1-PLAN-LOTE-845] Redis falló anotando un fallo de revisión; memoria local: {e}")
            total = None
    if total is None:
        with _cerrojo:
            _local_purgar(ahora)
            _fallos_locales.append(ahora)
            total = len(_fallos_locales)
    # Una alerta al cruzar cada umbral (no en cada fallo): al pasar de 10 y al llegar a 30, que es cuando bloquea.
    if total == UMBRAL_ALERTA + 1:
        _alerta_fallos(total, bloqueado=False)
    elif total == UMBRAL_BLOQUEO:
        _alerta_fallos(total, bloqueado=True)


def _avisar_bloqueo() -> None:
    """Un warning por ventana, no por intento."""
    global _bloqueo_avisado_hasta
    ahora = time.time()
    if ahora < _bloqueo_avisado_hasta:
        return
    _bloqueo_avisado_hasta = ahora + VENTANA_S
    logger.warning(f"🛡 [P1-PLAN-LOTE-845] Acceso de App Review BLOQUEADO: {UMBRAL_BLOQUEO}+ fallos en una hora con el "
                   "correo de demostración; el código fijo no vale hasta que la ventana se vacíe.")


def _alerta_fallos(total: int, bloqueado: bool) -> None:
    alert_key = "review_login_failed_burst"
    severidad = "critical" if bloqueado else "warning"
    message = (
        f"{total} intentos fallidos en una hora con el correo de demostración de App Review. "
        + ("El código fijo queda BLOQUEADO hasta que la ventana se vacíe (el revisor tampoco puede entrar). "
           if bloqueado else "")
        + f"Si no es un revisor equivocándose, alguien prueba códigos: rota {ENV_HASH} y valora cambiar el correo."
    )
    logger.warning(f"🛡 [P1-PLAN-LOTE-845] {total} fallos/hora con el correo de App Review (bloqueado={bloqueado}).")
    try:
        execute_sql_write(
            """
            INSERT INTO system_alerts
                (alert_key, alert_type, severity, title, message, metadata, affected_user_ids, triggered_at, resolved_at)
            VALUES (%s, %s, %s, %s, %s, %s::jsonb, '[]'::jsonb, NOW(), NULL)
            ON CONFLICT (alert_key) DO UPDATE
            SET severity = EXCLUDED.severity, title = EXCLUDED.title, message = EXCLUDED.message,
                metadata = EXCLUDED.metadata, triggered_at = EXCLUDED.triggered_at, resolved_at = NULL
            """,
            (alert_key, "review_login_failed_burst", severidad, "Intentos fallidos contra el acceso de App Review",
             message, json.dumps({"failures_last_hour": total, "locked": bloqueado}, ensure_ascii=False)),
        )
    except Exception as e:
        logger.error(f"❌ [P1-PLAN-LOTE-845] No se pudo persistir la alerta {alert_key}: {type(e).__name__}: {e}")


def cuenta_existente(correo: str) -> Optional[dict]:
    """La cuenta de demostración tal como está en `neon_auth."user"`, o None si no existe o está vetada.

    Solo lectura: este camino JAMÁS da de alta una identidad (un código fijo no puede ser una puerta de registro)."""
    filas = execute_sql_query(
        'SELECT id::text AS id, email, name, banned FROM neon_auth."user" WHERE lower(email) = %s LIMIT 1',
        (str(correo or "").strip().lower(),), fetch_all=True,
    )
    if not filas:
        return None
    fila = filas[0]
    if fila.get("banned") or not fila.get("id"):
        return None
    return fila


def anotar_uso(user_id: str) -> None:
    """Rastro de un uso con sesión emitida: `logger.warning` + `review_login_used:<user_id>` en `system_alerts`.

    Una fila por cuenta: `metadata.uses` suma cada uso, `first_used_at` se conserva y cada uso reabre la alerta
    (`resolved_at = NULL`). Best-effort como el resto de emisores: la alerta que no se pudo escribir no le quita la
    sesión al revisor (el `logger.warning` ya quedó). tooltip-anchor: P1-PLAN-LOTE-845-RASTRO"""
    uid = str(user_id)
    logger.warning(f"🔑 [P1-PLAN-LOTE-845] Acceso de App Review usado: sesión emitida con el código fijo (uid={uid[:8]}…).")
    ahora = datetime.now(timezone.utc).isoformat()
    alert_key = f"review_login_used:{uid}"
    metadata = {"user_id": uid, "uses": 1, "first_used_at": ahora, "last_used_at": ahora}
    message = (
        f"Se entró en la cuenta de demostración {uid[:8]} con el código fijo de App Review. Si no fue un revisor de "
        f"Apple, el código se filtró: cámbialo ({ENV_HASH}) o apaga el acceso quitando las dos variables."
    )
    try:
        execute_sql_write(
            """
            INSERT INTO system_alerts
                (alert_key, alert_type, severity, title, message, metadata, affected_user_ids, triggered_at, resolved_at)
            VALUES (%s, %s, 'warning', %s, %s, %s::jsonb, %s::jsonb, NOW(), NULL)
            ON CONFLICT (alert_key) DO UPDATE
            SET severity = EXCLUDED.severity, title = EXCLUDED.title, message = EXCLUDED.message,
                metadata = EXCLUDED.metadata || jsonb_build_object(
                    'uses', COALESCE((system_alerts.metadata->>'uses')::int, 0) + 1,
                    'first_used_at', COALESCE(system_alerts.metadata->>'first_used_at',
                                              EXCLUDED.metadata->>'first_used_at')),
                affected_user_ids = EXCLUDED.affected_user_ids,
                triggered_at = EXCLUDED.triggered_at, resolved_at = NULL
            """,
            (alert_key, "review_login_used", "Acceso de App Review usado", message,
             json.dumps(metadata, ensure_ascii=False), json.dumps([uid])),
        )
    except Exception as e:
        logger.error(f"❌ [P1-PLAN-LOTE-845] No se pudo persistir la alerta {alert_key}: {type(e).__name__}: {e}")


def anotar_cuenta_ausente() -> None:
    """El código era el correcto pero la cuenta no existe (o está vetada): el revisor no podrá entrar hasta que se
    vuelva a crear. Pasa si un revisor probó el borrado de cuenta con la de demostración. Best-effort."""
    logger.error(
        "🛑 [P1-PLAN-LOTE-845] Código de App Review correcto pero la cuenta de demostración no existe (o está "
        "vetada): no se crea; 401 como un código inválido. Hay que volver a crearla."
    )
    ahora = datetime.now(timezone.utc).isoformat()
    alert_key = "review_login_account_missing"
    message = (
        "Alguien escribió el código fijo de App Review, pero la cuenta de demostración no existe o está vetada, así "
        "que no pudo entrar. Vuelve a crearla (entrando una vez con el código de Neon) antes de la próxima revisión."
    )
    try:
        execute_sql_write(
            """
            INSERT INTO system_alerts
                (alert_key, alert_type, severity, title, message, metadata, affected_user_ids, triggered_at, resolved_at)
            VALUES (%s, %s, 'critical', %s, %s, %s::jsonb, '[]'::jsonb, NOW(), NULL)
            ON CONFLICT (alert_key) DO UPDATE
            SET severity = EXCLUDED.severity, title = EXCLUDED.title, message = EXCLUDED.message,
                metadata = EXCLUDED.metadata, triggered_at = EXCLUDED.triggered_at, resolved_at = NULL
            """,
            (alert_key, "review_login_account_missing", "Cuenta de App Review inexistente", message,
             json.dumps({"last_attempt_at": ahora}, ensure_ascii=False)),
        )
    except Exception as e:
        logger.error(f"❌ [P1-PLAN-LOTE-845] No se pudo persistir la alerta {alert_key}: {type(e).__name__}: {e}")


def cuenta_privilegiada(user_id: str) -> bool:
    """[ronda 1] La cuenta de demostración NO puede ser una cuenta con privilegios: un código fijo de 6 cifras abriría el
    panel de administración. Cuenta como privilegiada si `admin_acceso.es_admin` la reconoce O si está en la lista de
    admins aunque el panel esté apagado hoy (la sesión dura 30 días y el panel se puede encender mañana). Si no se puede
    comprobar, se trata como privilegiada (fail-secure)."""
    try:
        import admin_acceso
        uid = str(user_id or "").strip().lower()
        return bool(admin_acceso.es_admin(uid) or uid in admin_acceso.admin_ids())
    except Exception as e:
        logger.error(f"❌ [P1-PLAN-LOTE-845] no se pudo comprobar si la cuenta de revisión es admin: {type(e).__name__}")
        return True


def anotar_cuenta_privilegiada(user_id: str) -> None:
    """El código fijo llevaba a una cuenta de administración: no se emite sesión (401 como un código inválido)."""
    uid = str(user_id)
    logger.error(f"🛑 [P1-PLAN-LOTE-845] El correo de App Review es de una cuenta ADMIN (uid={uid[:8]}…): sin sesión.")
    alert_key = "review_login_account_privileged"
    message = (
        f"El correo configurado en {ENV_CORREO} es de una cuenta de administración ({uid[:8]}). El código fijo NO abre "
        "sesiones de admin: usa una cuenta de demostración sin privilegios, o quita esa cuenta de MEALFIT_ADMIN_USER_IDS."
    )
    try:
        execute_sql_write(
            """
            INSERT INTO system_alerts
                (alert_key, alert_type, severity, title, message, metadata, affected_user_ids, triggered_at, resolved_at)
            VALUES (%s, %s, 'critical', %s, %s, %s::jsonb, %s::jsonb, NOW(), NULL)
            ON CONFLICT (alert_key) DO UPDATE
            SET severity = EXCLUDED.severity, title = EXCLUDED.title, message = EXCLUDED.message,
                metadata = EXCLUDED.metadata, affected_user_ids = EXCLUDED.affected_user_ids,
                triggered_at = EXCLUDED.triggered_at, resolved_at = NULL
            """,
            (alert_key, "review_login_account_privileged", "El acceso de App Review apunta a una cuenta admin", message,
             json.dumps({"user_id": uid, "last_attempt_at": datetime.now(timezone.utc).isoformat()}, ensure_ascii=False),
             json.dumps([uid])),
        )
    except Exception as e:
        logger.error(f"❌ [P1-PLAN-LOTE-845] No se pudo persistir la alerta {alert_key}: {type(e).__name__}: {e}")
