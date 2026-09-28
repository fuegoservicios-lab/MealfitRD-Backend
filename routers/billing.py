import asyncio
import calendar
import os
import json
import logging
import time
import uuid
from datetime import datetime, timezone
import httpx
from fastapi import APIRouter, Body, Depends, HTTPException, Request
from error_utils import safe_error_detail
from typing import Optional

# Imports relativos al backend
from auth import get_verified_user_id
from db import execute_sql_query, execute_sql_write
from knobs import _env_float, _env_bool, _env_str, is_production
from psycopg.types.json import Jsonb
from rate_limiter import RateLimiter


# [P1-ASYNC-SYNC-DB-BLOCKING · 2026-05-24] Helper SSOT para despachar calls
# DB sync desde handlers `async def`. Pre-fix los 6 callsites DB corrían
# inline dentro del event loop, bloqueando ~10-200ms por roundtrip y
# throttlando otros handlers async bajo carga (chat stream, webhook PayPal,
# diary upload). Mismo patrón que P2-AUTH-ASYNC-SLEEP cerró para
# `auth.py::get_verified_user_id`. `asyncio.to_thread` despacha al default
# thread pool — el event loop sirve otras requests mientras Postgres responde.
# [P1-NEON-DB-MIGRATION · 2026-06-12] Los thunks envuelven
# `execute_sql_query`/`execute_sql_write` (SQL directo a Neon).
# Tooltip-anchor: P1-ASYNC-SYNC-DB-BLOCKING.
async def _run_sync_db_in_thread(thunk):
    return await asyncio.to_thread(thunk)

logger = logging.getLogger(__name__)

# [P1-NEW-HTTPX-TIMEOUT · 2026-05-15] Timeout obligatorio para los 4 callsites
# httpx a PayPal API (OAuth + subscription get + cancel + webhook verify).
# Pre-fix instanciaba el AsyncClient sin parámetro `timeout=` explícito;
# httpx default es Timeout(timeout=5.0) SOLO para el connect — el pool default
# tiene `read=None` en versiones antiguas, dejando reads colgados
# indefinidamente. Bajo tail-latency PayPal
# (incidente regional en `api-m.paypal.com`, 2026-04 observado externamente),
# `await client.post(...)` cuelga el worker FastAPI → pool exhausted → cascada
# 503. El knob permite a SRE bumpear/recortar sin redeploy si PayPal entra en
# degradación sostenida. Default 15s = mismo que `chat.py:261` (patrón
# pre-existente). Clamp [5, 60]: 5s mínimo para no fallar requests legítimos
# en latencia normal, 60s máximo para no permitir overrides absurdos que
# anulen la defensa. Tooltip-anchor: P1-NEW-HTTPX-TIMEOUT.
_HTTPX_TIMEOUT_S = _env_float(
    "MEALFIT_HTTPX_TIMEOUT_S",
    15.0,
    validator=lambda v: 5.0 <= v <= 60.0,
)

# [P1-BILLING-3 · 2026-05-12] Rate-limiter para /api/discount/validate.
# Pre-fix: endpoint público sin auth ni throttle → atacante brute-force
# de la tabla `discount_codes` enumerando códigos válidos por chunks
# alfabéticos sin coste. 20 calls/min/user es generoso para UX legítima
# (probar 1-2 códigos por sesión) y bloqueante para enumeración. Anchor:
# P1-BILLING-3-DISCOUNT-RATELIMIT.
_DISCOUNT_VALIDATE_LIMITER = RateLimiter(max_calls=20, period_seconds=60)

# [P2-RATELIMIT-COVERAGE · 2026-05-12] Rate-limiter para `/api/webhooks/paypal`.
# El webhook verifica firma criptográfica con PayPal (cierre del audit
# original P0), pero el flow de verify es CARO: 1 OAuth POST + 1 verify POST
# a `api-m.paypal.com` por cada request, ANTES de validar la firma. Un
# atacante anónimo flooding `/api/webhooks/paypal` con bodies arbitrarios
# consume ~2 round-trips httpx + tokens PayPal por cada request hasta que
# la firma falla. 30 calls/min/IP es generoso para PayPal real (2-3 eventos
# legítimos/min en peak) y bloquea floods costosos pre-signature-check.
# Anchor: P2-RATELIMIT-COVERAGE-PAYPAL-WEBHOOK.
_PAYPAL_WEBHOOK_LIMITER = RateLimiter(max_calls=30, period_seconds=60)

# [P1-PLAN-LOTE-714 · 2026-09-28] Limitadores de la tarjeta «Suscripción» de Configuración. Ninguno de los dos
# endpoints gasta IA, así que NO van por el paywall (`verify_api_quota`): al tope de créditos el usuario tiene que
# poder VER su suscripción y CANCELARLA — un 402 ahí lo dejaría pagando sin salida. El freno anti-abuso es el
# RateLimiter. `/cancel` va corto (5/min): cada llamada son hasta cuatro viajes a PayPal, y una persona de verdad
# cancela una vez. `/status` admite 30/min: la tarjeta lo pide al abrirse y tras cada acción.
_SUBSCRIPTION_CANCEL_LIMITER = RateLimiter(max_calls=5, period_seconds=60)
_SUBSCRIPTION_STATUS_LIMITER = RateLimiter(max_calls=30, period_seconds=60)

# [P1-PLAN-LOTE-714 · 2026-09-28] `/status` es una LECTURA para pintar una tarjeta: si PayPal tarda, se responde con
# el perfil y `paypal_status: null` en vez de colgar la pantalla los 15 s de `_HTTPX_TIMEOUT_S` (que sigue siendo el
# techo del cliente; este es el tope por petición). La caché por suscripción evita pedirle a PayPal lo mismo en cada
# visita a Configuración.
_STATUS_PAYPAL_TIMEOUT_S = _env_float(
    "MEALFIT_BILLING_STATUS_PAYPAL_TIMEOUT_S",
    4.0,
    validator=lambda v: 1.0 <= v <= 15.0,
)
_STATUS_PAYPAL_CACHE_TTL_S = _env_float(
    "MEALFIT_BILLING_STATUS_CACHE_TTL_S",
    600.0,
    validator=lambda v: 0.0 <= v <= 3600.0,
)


# ---------------------------------------------------------------------------
# [P1-BILLING-UPGRADE-FAIL-LOUD + P1-BILLING-CANCEL-FAIL-LOUD · 2026-05-12]
# Helpers para tratar respuestas de PayPal `/cancel` como idempotentes
# cuando indican que la sub ya estaba cancelada/inexistente, y para
# persistir alerts en `system_alerts` cuando el cancel real falla.
#
# Modo de fallo pre-fix (audit production-readiness 2026-05-12):
#   1. `/verify` (upgrade path lines ~276-288): si la cancel de la sub
#      vieja respondía != 204, solo `logger.warning` y seguía con el
#      UPDATE de plan_tier. PayPal seguía cobrando la sub vieja en
#      paralelo con la nueva → DOBLE COBRO silencioso al cliente hasta
#      intervención manual.
#   2. `/cancel` (lines ~370-371): si PayPal /cancel respondía != 204/200,
#      solo `logger.error` y seguía con UPDATE de subscription_status =
#      CANCELLED. Cliente veía "cancelado" en la UI pero PayPal seguía
#      cobrando → estado divergente sin alert.
#
# Fix: fail-loud antes de mutar BD, con tratamiento idempotente del caso
# "sub ya estaba cancelada" (404 o 422 con issue específico).
# Anchors: P1-BILLING-UPGRADE-FAIL-LOUD, P1-BILLING-CANCEL-FAIL-LOUD.
# ---------------------------------------------------------------------------

# Issues que PayPal devuelve cuando la sub ya está cancelada/expirada/inactiva.
# Tratarlos como idempotent success: el efecto deseado (no-billing futuro) ya
# ocurrió, no es un error real.
_PAYPAL_ALREADY_CANCELLED_ISSUES = (
    "SUBSCRIPTION_STATUS_INVALID",
    "INVALID_SUBSCRIPTION_STATUS",
    "SUBSCRIPTION_ALREADY_CANCELLED",
)


def _is_paypal_cancel_idempotent_success(status_code: int, body_text: str) -> bool:
    """True cuando PayPal indica que la sub ya estaba cancelada/inexistente.

    Casos cubiertos:
      - 200 / 204: cancel ejecutado ahora.
      - 404: la sub no existe (ya fue purgada). Para nosotros, "no-billing
        futuro" se cumple → idempotente.
      - 422 con `details[].issue` ∈ `_PAYPAL_ALREADY_CANCELLED_ISSUES`: la
        sub existe pero está en estado terminal (cancelled/expired/suspended).

    Cualquier otro status (5xx, 401, 422 con otro issue) NO es idempotente
    — el caller debe fail-loud para evitar estado divergente con PayPal.
    """
    if status_code in (200, 204):
        return True
    if status_code == 404:
        return True
    if status_code == 422:
        try:
            body = json.loads(body_text or "{}")
            for d in (body.get("details") or []):
                issue = (d.get("issue") or "").upper()
                if issue in _PAYPAL_ALREADY_CANCELLED_ISSUES:
                    return True
        except Exception:
            # Body no-JSON o malformado — NO asumir idempotencia.
            return False
    return False


def _persist_billing_alert(
    *,
    alert_key: str,
    severity: str,
    title: str,
    message: str,
    metadata: dict | None = None,
) -> None:
    """Best-effort UPSERT a `system_alerts` con `alert_type='billing'`.

    Idempotente vía `ON CONFLICT (alert_key)`: alerts repetidos del mismo
    `(user_id, sub_id)` actualizan `triggered_at` sin duplicar filas.
    Cualquier excepción se loguea sin propagar — la alert es observabilidad
    y NO debe romper el flujo del handler que ya está mid-failure.

    Modelo de resolution: **Manual** (ver "Política de `system_alerts`
    resolution" en CLAUDE.md). SRE debe verificar PayPal dashboard y
    reconciliar BD manualmente; no hay auto-cierre.
    """
    try:
        from datetime import datetime, timezone
        now_utc = datetime.now(timezone.utc)
        # [P1-NEON-DB-MIGRATION · 2026-06-12] SQL directo (antes PostgREST
        # upsert). El DO UPDATE reescribe todas las columnas del payload,
        # incl. `resolved_at = NULL` — re-emitir el mismo alert_key "reabre"
        # la alert (parity con el upsert PostgREST previo).
        execute_sql_write(
            """
            INSERT INTO public.system_alerts
                (alert_key, alert_type, severity, title, message, metadata, triggered_at, resolved_at)
            VALUES (%s, 'billing', %s, %s, %s, %s, %s, NULL)
            ON CONFLICT (alert_key) DO UPDATE
            SET alert_type = EXCLUDED.alert_type,
                severity = EXCLUDED.severity,
                title = EXCLUDED.title,
                message = EXCLUDED.message,
                metadata = EXCLUDED.metadata,
                triggered_at = EXCLUDED.triggered_at,
                resolved_at = NULL
            """,
            (alert_key, severity, title, message, Jsonb(metadata or {}), now_utc),
        )
    except Exception as e:
        logger.error(
            f"[P1-BILLING-FAIL-LOUD] No se pudo persistir alert "
            f"{alert_key}: {type(e).__name__}: {e}"
        )


# ---------------------------------------------------------------------------
# [P1-PLAN-LOTE-714 · 2026-09-28] Hasta cuándo pagó el usuario.
#
# EL AGUJERO: al cancelar (por `/cancel` o por el webhook `BILLING.SUBSCRIPTION.CANCELLED`), si PayPal no traía
# `billing_info.next_billing_time` se escribía SOLO `subscription_status = 'CANCELLED'`. El cargador del perfil
# (`db_profiles.get_user_profile`, P1-PROD-AUDIT-3) degrada a `gratis` todo CANCELLED sin fecha de fin — con razón:
# sin fecha no hay base para conceder gracia —, así que quien había pagado el mes entero perdía el plan AL INSTANTE.
# Y el webhook de una cancelación hecha desde la cuenta de PayPal casi nunca trae `next_billing_time` (una
# suscripción cancelada ya no tiene próximo cobro): era el caso NORMAL, no el raro.
#
# EL CIERRE: la fecha pagada se deriva de lo que sí está, `billing_info.last_payment.time` + UN intervalo del plan
# (mensual o anual, por el MISMO mapeo de env vars `PAYPAL_PLAN_*_ID` / `*_ANNUAL_ID` que da el tier). Nunca más de lo
# pagado: un solo intervalo desde el último cobro; si el `plan_id` figura en las dos listas, el intervalo corto; y si
# la fila ya estaba CANCELLED con fecha (la escribió `/cancel` con el `next_billing_time` de PayPal), se queda la
# MENOR (`LEAST`) — el cobro se captura minutos u horas después del inicio del ciclo, así que «último pago + 1 mes»
# tiende a pasarse un poco del ciclo real. Solo si no se puede saber nada se conserva la conducta de antes (degradar
# ya) y se alerta `billing_cancel_access_end_unknown:<user>:<sub>`, para que un operador devuelva los días pagados.
# tooltip-anchor: P1-PLAN-LOTE-714-ACCESS-UNTIL
# ---------------------------------------------------------------------------

#: [P1-PLAN-LOTE-714] Estados del perfil en los que la suscripción guardada sigue cobrando (o está a punto). Pisar su
#: handle sin cancelarla en PayPal es el doble cobro de P1-BILLING-UPGRADE-FAIL-LOUD; cualquier otro estado
#: (CANCELLED, INACTIVE, NULL…) es una suscripción que ya no cobra.
_LIVE_SUBSCRIPTION_STATUSES = frozenset({"ACTIVE", "PAYMENT_RETRYING", "APPROVED"})
_LIVE_SUBSCRIPTION_STATUSES_SQL = ", ".join(f"'{s}'" for s in sorted(_LIVE_SUBSCRIPTION_STATUSES))

_CANCEL_OK_MESSAGE = (
    "Tu suscripción no se renovará, pero mantendrás tu plan actual hasta el final del ciclo pagado."
)
_CANCEL_OK_SIN_FECHA_MESSAGE = "Tu suscripción fue cancelada y no se renovará."


def _parse_paypal_time(value) -> Optional[datetime]:
    """Instante ISO-8601 (de PayPal o de Postgres) o `datetime` → `datetime` en UTC; None si falta o no se entiende."""
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str) and value.strip():
        try:
            dt = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _as_utc_iso(value) -> Optional[str]:
    dt = _parse_paypal_time(value)
    return dt.isoformat() if dt is not None else None


def _is_future(value) -> bool:
    dt = _parse_paypal_time(value)
    return dt is not None and dt > datetime.now(timezone.utc)


def _add_months(dt: datetime, months: int) -> datetime:
    """Meses de calendario; el día se recorta al último del mes destino (31-ene + 1 mes = 28/29-feb)."""
    idx = dt.month - 1 + months
    year, month = dt.year + idx // 12, idx % 12 + 1
    return dt.replace(year=year, month=month, day=min(dt.day, calendar.monthrange(year, month)[1]))


def _plan_interval_months(plan_id) -> Optional[int]:
    """1 (mensual) | 12 (anual) | None si el `plan_id` no es de ninguno. MISMAS env vars que
    `_build_paypal_plan_tier_map`: un plan que no da tier tampoco da intervalo."""
    if not plan_id:
        return None
    mensual = anual = False
    for tier in ("BASIC", "PLUS", "ULTRA"):
        mensual = mensual or os.environ.get(f"PAYPAL_PLAN_{tier}_ID") == plan_id
        anual = anual or os.environ.get(f"PAYPAL_PLAN_{tier}_ANNUAL_ID") == plan_id
    if mensual:
        return 1  # también si el mismo id figura como anual: ante la duda, el período corto
    return 12 if anual else None


def _paid_through(sub_data) -> tuple:
    """`(fin del período pagado, fuente)` de un recurso de suscripción de PayPal.

    Fuente: `'next_billing_time'` (lo dice PayPal), `'last_payment'` (último cobro + un intervalo del plan) o
    `'indeterminada'` (fin = None)."""
    if not isinstance(sub_data, dict):
        return None, "indeterminada"
    billing_info = sub_data.get("billing_info")
    billing_info = billing_info if isinstance(billing_info, dict) else {}
    next_billing = _parse_paypal_time(billing_info.get("next_billing_time"))
    if next_billing is not None:
        return next_billing, "next_billing_time"
    last_payment = billing_info.get("last_payment")
    last_payment = last_payment if isinstance(last_payment, dict) else {}
    pagado_en = _parse_paypal_time(last_payment.get("time"))
    meses = _plan_interval_months(sub_data.get("plan_id"))
    if pagado_en is not None and meses:
        return _add_months(pagado_en, meses), "last_payment"
    return None, "indeterminada"


def _json_object_or_none(resp) -> Optional[dict]:
    try:
        data = resp.json()
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


async def _fetch_paypal_subscription(paypal_api_base, access_token, subscription_id) -> Optional[dict]:
    """GET best-effort de la suscripción en PayPal. None si no hay token, si PayPal no responde 200 o si el cuerpo no es
    un objeto JSON. No lanza: cada caller ya tiene su camino para «no se sabe» (y lo deja en el log)."""
    if not (paypal_api_base and access_token and subscription_id):
        return None
    try:
        async with httpx.AsyncClient(timeout=_HTTPX_TIMEOUT_S) as client:
            resp = await client.get(
                f"{paypal_api_base}/v1/billing/subscriptions/{subscription_id}",
                headers={"Authorization": f"Bearer {access_token}"},
            )
    except httpx.HTTPError as e:
        logger.warning(f"[P1-PLAN-LOTE-714] GET de la sub {subscription_id} en PayPal falló: {type(e).__name__}: {e}")
        return None
    if resp.status_code != 200:
        logger.warning(f"[P1-PLAN-LOTE-714] GET de la sub {subscription_id} en PayPal respondió {resp.status_code}.")
        return None
    return _json_object_or_none(resp)


def _mark_subscription_cancelled(*, where_column: str, where_value, hasta: Optional[datetime]) -> list:
    """UPDATE a CANCELLED (sync: el caller lo despacha con `_run_sync_db_in_thread`). Devuelve las filas tocadas,
    `[{id, subscription_end_date}]`, con la fecha que QUEDÓ — la que el cargador del perfil va a honrar.

    Con `hasta`: si la fila ya estaba CANCELLED con fecha propia se queda la MENOR (el webhook que llega después de
    `/cancel` no alarga lo que `/cancel` escribió). Sin `hasta`: solo el estado, como antes; la fecha que hubiera se
    conserva."""
    if where_column not in ("id", "paypal_subscription_id"):
        raise ValueError(f"columna de filtro no permitida: {where_column!r}")
    if hasta is not None:
        fin = hasta.isoformat()
        sql = (
            "UPDATE public.user_profiles "
            "   SET subscription_status = 'CANCELLED', "
            "       subscription_end_date = CASE "
            "           WHEN subscription_status = 'CANCELLED' AND subscription_end_date IS NOT NULL "
            "           THEN LEAST(subscription_end_date, %s::timestamptz) "
            "           ELSE %s::timestamptz END "
            f" WHERE {where_column} = %s "
            "RETURNING id, subscription_end_date"
        )
        params = (fin, fin, where_value)
    else:
        sql = (
            "UPDATE public.user_profiles SET subscription_status = 'CANCELLED' "
            f" WHERE {where_column} = %s "
            "RETURNING id, subscription_end_date"
        )
        params = (where_value,)
    filas = execute_sql_write(sql, params, returning=True)
    return filas if isinstance(filas, list) else []


def _alert_access_end_unknown(*, filas: list, subscription_id, origen: str) -> None:
    """Alerta (sync) por cada fila cancelada que se queda SIN fecha de fin futura: el cargador del perfil la degradará
    ya, aunque el usuario quizá pagó el período en curso. Una fila con fecha futura (la escribió `/cancel`) no alerta:
    no hay nada que devolver."""
    for fila in filas:
        fin_bd = _as_utc_iso(fila.get("subscription_end_date"))
        if _is_future(fin_bd):
            continue
        uid = str(fila.get("id"))
        _persist_billing_alert(
            alert_key=f"billing_cancel_access_end_unknown:{uid}:{subscription_id}",
            severity="warning",
            title="Cancelación: no se supo hasta cuándo pagó — el plan se degrada ya",
            message=(
                f"User {uid}: la sub {subscription_id} quedó CANCELLED ({origen}) sin que PayPal diera "
                f"next_billing_time ni un último pago con un plan mapeable. Fecha de fin en BD: "
                f"{fin_bd or 'ninguna'}, así que el cargador del perfil la degrada a gratis. Verificar en PayPal el "
                f"último cobro y, si el período pagado sigue vigente, fijar subscription_end_date y restaurar plan_tier."
            ),
            metadata={
                "user_id": uid, "sub_id": str(subscription_id), "origen": origen,
                "subscription_end_date": fin_bd,
            },
        )
        logger.warning(
            f"⚠️ [P1-PLAN-LOTE-714] sub {subscription_id} de {uid} cancelada ({origen}) sin fecha de fin determinable."
        )


# ---------------------------------------------------------------------------
# [P1-PLAN-LOTE-714 · 2026-09-28] Estado VIVO de la suscripción en PayPal para `GET /api/subscription/status`, en caché
# de proceso por id de suscripción. El id sale siempre de la fila del PROPIO usuario (filtrada por el id del JWT), así
# que la caché no puede servirle a nadie el estado de otro. Solo guarda `{status, next_billing_time}`: ni el payload
# crudo (email del comprador, dirección) ni tokens. Un fallo también se cachea, pero corto: con PayPal caído la
# tarjeta no debe martillarlo en cada visita. Cualquier webhook, `/cancel` o `/verify` de esa sub la invalida (en
# este proceso; los demás workers la ven caducar por TTL).
# tooltip-anchor: P1-PLAN-LOTE-714-STATUS-CACHE
# ---------------------------------------------------------------------------
_PAYPAL_STATUS_CACHE: dict = {}
_PAYPAL_STATUS_CACHE_MAX = 4096
_PAYPAL_STATUS_NEG_TTL_S = 60.0


def _invalidate_paypal_status_cache(*subscription_ids) -> None:
    for sid in subscription_ids:
        if sid:
            _PAYPAL_STATUS_CACHE.pop(str(sid), None)


async def _paypal_status_snapshot(subscription_id: str) -> Optional[dict]:
    """`{status, next_billing_time}` de la sub según PayPal, o None si PayPal no está configurado o no respondió.
    Fail-soft (es una LECTURA): no lanza; deja un warning en el log."""
    ahora = time.monotonic()
    cacheado = _PAYPAL_STATUS_CACHE.get(subscription_id)
    if cacheado is not None and cacheado[0] > ahora:
        return cacheado[1]

    snapshot = None
    client_id = os.environ.get("PAYPAL_CLIENT_ID")
    secret = os.environ.get("PAYPAL_SECRET")
    if not (client_id and secret):
        logger.warning("[P1-PLAN-LOTE-714] /status: PayPal sin credenciales; se responde solo con el perfil.")
    else:
        api_base = "https://api-m.sandbox.paypal.com" if not is_production() else "https://api-m.paypal.com"

        async def _consultar() -> Optional[dict]:
            async with httpx.AsyncClient(timeout=_HTTPX_TIMEOUT_S) as client:
                auth_resp = await client.post(
                    f"{api_base}/v1/oauth2/token",
                    auth=(client_id, secret),
                    data={"grant_type": "client_credentials"},
                    timeout=_STATUS_PAYPAL_TIMEOUT_S,
                )
                if auth_resp.status_code != 200:
                    logger.warning(f"[P1-PLAN-LOTE-714] /status: OAuth de PayPal respondió {auth_resp.status_code}.")
                    return None
                token = (_json_object_or_none(auth_resp) or {}).get("access_token")
                sub_resp = await client.get(
                    f"{api_base}/v1/billing/subscriptions/{subscription_id}",
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=_STATUS_PAYPAL_TIMEOUT_S,
                )
            datos = _json_object_or_none(sub_resp) if sub_resp.status_code == 200 else None
            if datos is None:
                logger.warning(
                    f"[P1-PLAN-LOTE-714] /status: GET de la sub {subscription_id} respondió "
                    f"{sub_resp.status_code} sin un objeto utilizable."
                )
                return None
            billing_info = datos.get("billing_info")
            billing_info = billing_info if isinstance(billing_info, dict) else {}
            return {
                "status": str(datos.get("status") or "").upper() or None,
                "next_billing_time": _as_utc_iso(billing_info.get("next_billing_time")),
            }

        try:
            # El timeout de httpx acota cada fase (conectar, leer…), no el total: el `wait_for` es el tope real de
            # lo que la tarjeta espera a PayPal antes de pintarse solo con el perfil.
            snapshot = await asyncio.wait_for(_consultar(), timeout=_STATUS_PAYPAL_TIMEOUT_S)
        except Exception as e:  # noqa: BLE001 — lectura fail-soft: cualquier fallo de PayPal cae al perfil
            logger.warning(
                f"[P1-PLAN-LOTE-714] /status: PayPal no respondió para la sub {subscription_id}: "
                f"{type(e).__name__}: {e}"
            )
            snapshot = None

    ttl = _STATUS_PAYPAL_CACHE_TTL_S if snapshot is not None else min(_PAYPAL_STATUS_NEG_TTL_S, _STATUS_PAYPAL_CACHE_TTL_S)
    if ttl > 0:
        if len(_PAYPAL_STATUS_CACHE) >= _PAYPAL_STATUS_CACHE_MAX:
            for clave in [k for k, (caduca, _) in _PAYPAL_STATUS_CACHE.items() if caduca <= ahora]:
                _PAYPAL_STATUS_CACHE.pop(clave, None)
            if len(_PAYPAL_STATUS_CACHE) >= _PAYPAL_STATUS_CACHE_MAX:
                _PAYPAL_STATUS_CACHE.clear()
        _PAYPAL_STATUS_CACHE[subscription_id] = (time.monotonic() + ttl, snapshot)
    return snapshot


# ---------------------------------------------------------------------------
# [P1-BILLING-ORPHAN-RECOVERY · 2026-08-22] Adopción de un cobro cuyo `/verify`
# nunca llegó.
#
# EL AGUJERO QUE CIERRA
# `POST /api/subscription/verify` era el ÚNICO camino por el que una suscripción
# de PayPal se escribía en `user_profiles`, y lo dispara el NAVEGADOR desde
# `onApprove`. Si entre la aprobación y esa llamada se caía la red, el usuario
# cerraba la pestaña, o `/verify` devolvía 500/409, PayPal cobraba y el sistema no
# se enteraba: `paypal_subscription_id` quedaba NULL y los webhooks `ACTIVATED` /
# `PAYMENT.SALE.COMPLETED` filtran justo por esa columna → 0 filas, no-op
# silencioso. Cobro recurrente sin acceso, sin alerta y sin cron de
# reconciliación (auditoría 2026-08-22): la pérdida era permanente.
#
# LA OTRA MITAD vive en el frontend: `PaymentModal` estampa `custom_id: <user_id>`
# al crear la suscripción, y PayPal nos lo devuelve FIRMADO dentro del webhook.
# Sin esa mitad este helper no tiene a quién atribuir nada (y lo dice en la alerta).
#
# LO QUE NO CONCEDE
#   - El TIER sigue derivando del `plan_id` vía `_build_paypal_plan_tier_map`
#     (I-Billing-1). El `custom_id` dice A QUIÉN, jamás QUÉ.
#   - Si la sub YA está en algún perfil, no es huérfana: el UPDATE no matchó por
#     el guard (P1-BILLING-REACTIVATE-NOT-CANCELLED), y resucitar una CANCELLED
#     por esta vía sería la puerta trasera que ese fix cerró.
#   - Si el perfil destino ya tiene OTRA suscripción viva, NO la pisa: cambiar el
#     handle sin cancelar la vieja en PayPal es el doble cobro de
#     P1-BILLING-UPGRADE-FAIL-LOUD. Alerta y deja el caso al operador.
#   - [P1-PLAN-LOTE-714 · 2026-09-28] Tampoco adopta un cobro cuyo MONTO no cuadra:
#     antes la adopción se saltaba la verificación que `/verify` sí hace
#     (`_verify_subscription_amount`, `plan_overridden`), y el ataque del precio
#     sobrescrito desde la consola del navegador ($0.01 por Max, auditoría
#     2026-08-22 §1) tenía una puerta trasera: `/verify` lo rechazaba con 409… y el
#     webhook ACTIVATED, al no encontrar la sub en ningún perfil, lo adoptaba por
#     el `custom_id`. Ahora corre LA MISMA verificación, con la misma política
#     (sin cupón que pueda explicarlo, se bloquea; con uno posible, alerta y sigue).
#   - [P1-PLAN-LOTE-714] «OTRA suscripción viva» significa VIVA: ACTIVE,
#     PAYMENT_RETRYING o APPROVED en el perfil. Antes bastaba con que el perfil
#     tuviera CUALQUIER `paypal_subscription_id`, y el que volvía tras cancelar
#     —pagaba, cerraba la pestaña— se quedaba en gratis mientras PayPal le cobraba.
#     Una sub vieja CANCELLED/INACTIVE ya no cobra: se reemplaza su handle.
#
# Knob `MEALFIT_BILLING_ORPHAN_RECOVERY` (default True) = kill-switch.
# Tooltip-anchor: P1-BILLING-ORPHAN-RECOVERY.
# ---------------------------------------------------------------------------

def _looks_like_user_id(value) -> bool:
    """True solo si `value` parsea como UUID.

    Necesario ANTES de meterlo en `WHERE id = %s`: `user_profiles.id` es `uuid`,
    así que un `custom_id` basura propagaría `invalid input syntax for type uuid`
    → 503 → PayPal reintentando 25 veces contra un error determinista.
    """
    try:
        uuid.UUID(str(value).strip())
        return True
    except (ValueError, AttributeError, TypeError):
        return False


def _orphan_unrecoverable(*, subscription_id, custom_id, tier, plan_id, motivo: str) -> None:
    """Alerta crítica: PayPal activó una sub que NO se atribuyó a nadie (sync)."""
    _persist_billing_alert(
        alert_key=f"billing_orphan_subscription_unrecoverable:{subscription_id}",
        severity="critical",
        title="Webhook: suscripción activa que no pertenece a nadie",
        message=(
            f"PayPal activó la sub {subscription_id} (plan {plan_id}) pero no se "
            f"atribuyó a ningún usuario: {motivo}. El cliente está pagando sin "
            f"recibir el plan — reconciliar a mano (el email del comprador está "
            f"en el evento de PayPal)."
        ),
        metadata={
            "sub_id": subscription_id, "plan_id": plan_id, "tier": tier,
            "custom_id": str(custom_id) if custom_id else None, "motivo": motivo,
        },
    )
    logger.error(
        f"❌ [P1-BILLING-ORPHAN-RECOVERY] sub {subscription_id} sin dueño: {motivo}"
    )


def _orphan_adoption_target(*, subscription_id, custom_id, tier, plan_id) -> Optional[tuple]:
    """¿A quién se puede atribuir la sub? `(user_id, sub_reemplazada_o_None)`, o None si no se adopta (con la alerta
    ya emitida, o en silencio si la sub tiene dueño). Sync: el caller lo despacha con `_run_sync_db_in_thread`."""

    def _unrecoverable(motivo: str) -> None:
        _orphan_unrecoverable(
            subscription_id=subscription_id, custom_id=custom_id, tier=tier, plan_id=plan_id, motivo=motivo,
        )

    if not tier:
        # Mismo criterio que `/verify`: un plan_id que no mapea no otorga nada.
        _unrecoverable(f"el plan_id {plan_id!r} no mapea a un tier interno")
        return None

    # ¿La tiene ya alguien? Entonces el UPDATE falló por el guard, no por orfandad.
    if execute_sql_query(
        "SELECT id FROM public.user_profiles WHERE paypal_subscription_id = %s",
        (subscription_id,),
        fetch_all=True,
    ):
        return None

    if not _looks_like_user_id(custom_id):
        _unrecoverable("el evento no trae un `custom_id` utilizable")
        return None

    uid = str(custom_id).strip()
    profile = execute_sql_query(
        "SELECT id, paypal_subscription_id, subscription_status FROM public.user_profiles WHERE id = %s",
        (uid,),
        fetch_one=True,
    )
    if not profile:
        _unrecoverable(f"el custom_id {uid} no corresponde a ningún perfil")
        return None
    sub_vieja = profile.get("paypal_subscription_id") or None
    estado_viejo = str(profile.get("subscription_status") or "").upper()
    # [P1-PLAN-LOTE-714] Solo una sub VIVA bloquea; una CANCELLED/INACTIVE ya no cobra y su handle se reemplaza.
    # tooltip-anchor: P1-PLAN-LOTE-714-ORPHAN-EXSUB
    if sub_vieja and estado_viejo in _LIVE_SUBSCRIPTION_STATUSES:
        _unrecoverable(
            f"el perfil {uid} ya tiene la sub {sub_vieja} viva ({estado_viejo}); "
            f"pisarla dejaría la vieja cobrando en PayPal sin handle"
        )
        return None
    return uid, sub_vieja


def _adopt_orphan_subscription(*, uid, sub_vieja, subscription_id, tier, plan_id) -> None:
    """UPDATE de adopción + alerta de auditoría (sync)."""
    # El `AND (...)` cierra la carrera entre la lectura y este UPDATE: si otro camino (un `/verify` que llegó tarde)
    # escribió entremedio una sub VIVA distinta, matchea 0 filas en vez de pisarla. Una sub vieja que ya no cobra
    # (CANCELLED/INACTIVE/NULL) no la protege: es el ex-suscriptor que vuelve (P1-PLAN-LOTE-714).
    # [P1-PLAN-LOTE-714] `subscription_end_date = NULL`: la fecha de fin de una cancelación ANTERIOR no pertenece a
    # esta suscripción; dejarla pintaba un «Próximo cobro» falso en Configuración.
    updated = execute_sql_write(
        "UPDATE public.user_profiles "
        "   SET subscription_status = %s, plan_tier = %s, paypal_subscription_id = %s, "
        "       subscription_end_date = NULL "
        " WHERE id = %s AND (paypal_subscription_id IS NULL OR paypal_subscription_id = %s "
        f"      OR upper(COALESCE(subscription_status, '')) NOT IN ({_LIVE_SUBSCRIPTION_STATUSES_SQL})) "
        "RETURNING id",
        ("ACTIVE", tier, subscription_id, uid, subscription_id),
        returning=True,
    )
    if not updated:
        _orphan_unrecoverable(
            subscription_id=subscription_id, custom_id=uid, tier=tier, plan_id=plan_id,
            motivo=f"carrera: el perfil {uid} cambió entre la lectura y el UPDATE",
        )
        return

    _persist_billing_alert(
        alert_key=f"billing_orphan_subscription_recovered:{uid}:{subscription_id}",
        severity="warning",
        title="Webhook: suscripción huérfana recuperada por custom_id",
        message=(
            f"La sub {subscription_id} (plan {plan_id} → tier {tier}) se activó en "
            f"PayPal sin que `/verify` llegara nunca. Se atribuyó al usuario {uid} "
            f"por el `custom_id` del evento"
            + (f", reemplazando su sub anterior {sub_vieja}, que ya no cobraba" if sub_vieja else "")
            + ". El tier ya está aplicado; revisar por qué falló el `/verify` del cliente."
        ),
        metadata={
            "user_id": uid, "sub_id": subscription_id, "plan_id": plan_id, "tier": tier,
            "sub_reemplazada": sub_vieja,
        },
    )
    logger.warning(
        f"⚙️ [P1-BILLING-ORPHAN-RECOVERY] sub {subscription_id} adoptada por "
        f"{uid} (tier {tier}) — el /verify de ese pago nunca llegó."
    )


async def _recover_orphan_subscription(
    *, subscription_id, custom_id, tier, plan_id, resource=None, access_token=None, paypal_api_base=None,
) -> None:
    """Atribuye una suscripción ACTIVE que no pertenece a ningún perfil, con la MISMA verificación de monto que
    `/verify` (P1-PLAN-LOTE-714). Los datos para esa verificación salen de un GET fresco a PayPal (lo mismo que mira
    `/verify`); si no se puede, del recurso del propio webhook, que viene firmado.
    tooltip-anchor: P1-PLAN-LOTE-714-ORPHAN-AMOUNT"""
    objetivo = await _run_sync_db_in_thread(
        lambda: _orphan_adoption_target(
            subscription_id=subscription_id, custom_id=custom_id, tier=tier, plan_id=plan_id,
        )
    )
    if not objetivo:
        return
    uid, sub_vieja = objetivo

    sub_data = resource if isinstance(resource, dict) else {}
    if access_token:
        fresca = await _fetch_paypal_subscription(paypal_api_base, access_token, subscription_id)
        if fresca:
            sub_data = fresca
    try:
        await _verify_subscription_amount(
            sub_data=sub_data,
            verified_plan_id=plan_id,
            tier=tier,
            coupon_code="",  # el webhook no trae el cupón: el caso «cupón no reenviado» de `/verify`
            access_token=access_token,
            paypal_api_base=paypal_api_base,
            user_id=uid,
            subscription_id=subscription_id,
        )
    except HTTPException as e:
        # `_verify_subscription_amount` ya dejó `billing_price_tampering:<uid>:<sub>` (critical). Esta segunda alerta
        # dice lo que el operador necesita además: que la sub NO se adoptó y sigue activa en PayPal.
        await _run_sync_db_in_thread(
            lambda: _orphan_unrecoverable(
                subscription_id=subscription_id, custom_id=custom_id, tier=tier, plan_id=plan_id,
                motivo=(
                    f"el monto cobrado no cuadra con el precio del plan (plan_overridden sin cupón que lo "
                    f"justifique; verificación de monto → {e.status_code}), así que NO se adopta para {uid}. "
                    f"Revisar la sub en PayPal y cancelarla o reembolsar"
                ),
            )
        )
        return
    await _run_sync_db_in_thread(
        lambda: _adopt_orphan_subscription(
            uid=uid, sub_vieja=sub_vieja, subscription_id=subscription_id, tier=tier, plan_id=plan_id,
        )
    )


router = APIRouter(
    prefix="/api/subscription",
    tags=["billing"],
)

# El webhook pasivo de paypal no puede tener el prefix /api/subscription si original era /api/webhooks/paypal
# Así que lo añadiremos directamente a la raíz de la app o podemos dejar que el router maneje '/api', pero el router ya tiene prefix.
# Vamos a crear otro router para webhooks o manejarlo aquí sin prefix.
webhooks_router = APIRouter(
    prefix="/api/webhooks",
    tags=["webhooks"]
)

discount_router = APIRouter(
    prefix="/api/discount",
    tags=["discount"],
)

@discount_router.post("/validate")
async def api_validate_discount(
    data: dict = Body(...),
    verified_user_id: Optional[str] = Depends(_DISCOUNT_VALIDATE_LIMITER),
):
    """Valida un código de descuento contra la tabla discount_codes en Postgres.

    [P1-BILLING-3 · 2026-05-12] Pre-fix el endpoint era público (sin
    `Depends(get_verified_user_id)`) y sin rate-limit. Un atacante anónimo
    podía enumerar `discount_codes` por brute-force alfabético sin coste —
    cada código válido encontrado era $X gratis para usar luego con
    `/verify`. Ahora:

      - `Depends(_DISCOUNT_VALIDATE_LIMITER)`: 20 calls/min/user (o IP si
        no autenticado). El limiter inyecta `verified_user_id` y bloquea
        bursts. Backed por Redis cuando disponible; in-memory fallback.
      - Auth obligatoria (`if not verified_user_id: 401`). Sin auth NO se
        valida — los descuentos solo se aplican a usuarios registrados.

    Anchor: P1-BILLING-3-DISCOUNT-RATELIMIT
    """
    if not verified_user_id:
        raise HTTPException(
            status_code=401,
            detail="Authentication required to validate discount codes.",
        )
    try:
        code = (data.get("code") or "").strip().upper()
        tier = (data.get("tier") or "").strip().lower()

        if not code:
            raise HTTPException(status_code=400, detail="Código requerido")

        # Buscar el código en la tabla.
        # [P1-NEON-DB-MIGRATION · 2026-06-12] `valid_from`/`valid_until` van
        # con `::text` para preservar el parseo `datetime.fromisoformat(...)`
        # de abajo (PostgREST devolvía strings ISO; psycopg devolvería datetime).
        rows = await _run_sync_db_in_thread(
            lambda: execute_sql_query(
                """
                SELECT discount_percent, max_uses, current_uses, applicable_tiers,
                       valid_from::text AS valid_from, valid_until::text AS valid_until
                  FROM public.discount_codes
                 WHERE code = %s AND is_active = TRUE
                """,
                (code,),
                fetch_all=True,
            )
        )

        if not rows or len(rows) == 0:
            return {"valid": False, "message": "Código no encontrado o inactivo."}

        discount = rows[0]

        # Verificar vigencia
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)

        if discount.get("valid_from"):
            valid_from = datetime.fromisoformat(discount["valid_from"].replace("Z", "+00:00"))
            if now < valid_from:
                return {"valid": False, "message": "Este código aún no está activo."}

        if discount.get("valid_until"):
            valid_until = datetime.fromisoformat(discount["valid_until"].replace("Z", "+00:00"))
            if now > valid_until:
                return {"valid": False, "message": "Este código ha expirado."}

        # Verificar usos
        if discount.get("max_uses") is not None:
            if discount.get("current_uses", 0) >= discount["max_uses"]:
                return {"valid": False, "message": "Este código ya alcanzó su límite de usos."}

        # Verificar tier aplicable
        applicable = discount.get("applicable_tiers", [])
        if applicable and tier and tier not in applicable:
            return {"valid": False, "message": f"Este código no aplica al plan seleccionado."}

        logger.info(f"✅ Código de descuento '{code}' validado: {discount['discount_percent']}% off")

        return {
            "valid": True,
            "discount_percent": discount["discount_percent"],
            "message": f"¡{discount['discount_percent']}% de descuento aplicado!"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error validando código de descuento: {str(e)}")
        raise HTTPException(status_code=500, detail="Error validando código")

# ---------------------------------------------------------------------------
# [P0-BILLING-1 · 2026-05-12] Mapping PayPal plan_id → tier interno.
# Construido server-side desde env vars. El cliente NO puede inyectar tier:
# pre-fix, el handler hacía `UPDATE user_profiles SET plan_tier = body.tier`
# confiando en el cliente → cualquier user con sub real de "basic" podía
# pasar tier="ultra" y obtener upgrade gratuito.
# Anchor: P0-BILLING-1-TIER-SERVER-SIDE.
#
# Si una env var falta, esa entrada se omite del mapping — verify_subscription
# rechazará el plan_id correspondiente con 400 ("Plan no reconocido").
# Operador debe setear las 3 env vars en producción.
# ---------------------------------------------------------------------------

def _build_paypal_plan_tier_map() -> dict[str, str]:
    # [P1-BILLING-ANNUAL-MAP · 2026-05-31] Mapear AMBAS variantes (mensual Y
    # anual) al mismo tier. El frontend (PaymentModal) usa 6 plan IDs distintos
    # (3 mensuales + 3 anuales, en VITE_PAYPAL_PLAN_{TIER}[_ANNUAL]). Pre-fix
    # este mapping solo leía los 3 mensuales `PAYPAL_PLAN_{TIER}_ID`; cuando un
    # usuario pagaba el plan ANUAL, PayPal devolvía el plan_id anual, que NO
    # mapeaba → HTTPException(400 "Plan no reconocido") DESPUÉS de que PayPal ya
    # cobró = cobro-sin-upgrade (limbo, chargeback/soporte). El operador DEBE
    # setear `PAYPAL_PLAN_{TIER}_ANNUAL_ID` en prod además de `PAYPAL_PLAN_{TIER}_ID`.
    # Cada env var ausente se omite (degradación segura: ese plan_id se rechaza).
    mapping: dict[str, str] = {}
    for tier in ("basic", "plus", "ultra"):
        for suffix in ("", "_ANNUAL"):
            plan_id = os.environ.get(f"PAYPAL_PLAN_{tier.upper()}{suffix}_ID")
            if plan_id:
                mapping[plan_id] = tier
    return mapping


# ---------------------------------------------------------------------------
# [P1-BILLING-AMOUNT · 2026-07-12] Verificación de INTEGRIDAD DE MONTO server-side.
#
# I-Billing-1 cierra el TIER (server-derived del plan_id) pero NO el MONTO cobrado.
# El cliente puede inyectar `plan.billing_cycles[].pricing_scheme.fixed_price` al
# crear la suscripción PayPal (PaymentModal lo hace legítimamente para descuentos con
# cupón) y cobrarse el PRIMER ciclo a un precio arbitrario, sin cupón válido. El tier
# no escala (server-side), pero el monto del ciclo 1 sí es manipulable.
#
# Defensa (invariante hermana I-Billing-AMOUNT): si PayPal marca `plan_overridden`,
# el precio fue sobrescrito → SOLO legítimo si un cupón válido (re-validado
# server-side) lo justifica. Knob de 3 modos (off/warn/block, default WARN por
# rollout seguro con caché PWA): warn detecta+alerta sin bloquear pagos legítimos;
# block además 409ea SOLO cuando hay underpayment PROBADO (precio parseable < mínimo
# esperado) — nunca bloquea el caso ambiguo (no-parseable / cupón no reenviado por
# frontend viejo), fail-cheap. Promover a `block` tras smoke-test de PayPal sandbox
# + días de warn limpio.
# ---------------------------------------------------------------------------

def _parse_price(value) -> Optional[float]:
    try:
        f = float(value)
        return f if f >= 0 else None
    except (TypeError, ValueError):
        return None


def _extract_regular_fixed_price(billing_cycles: list) -> Optional[float]:
    """Precio del ciclo REGULAR (o el primero) de una lista de billing_cycles PayPal."""
    try:
        chosen = None
        for c in (billing_cycles or []):
            if (c or {}).get("tenure_type") == "REGULAR":
                chosen = c
                break
        if chosen is None and billing_cycles:
            chosen = billing_cycles[0]
        if chosen:
            fp = ((chosen.get("pricing_scheme") or {}).get("fixed_price") or {})
            return _parse_price(fp.get("value"))
    except Exception:
        pass
    return None


def _extract_override_price(sub_data: dict) -> Optional[float]:
    """Best-effort: precio realmente pactado en la suscripción (plan embebido cuando
    `plan_overridden`, o el último pago capturado). None si no parseable → fail-cheap."""
    p = _extract_regular_fixed_price(((sub_data or {}).get("plan") or {}).get("billing_cycles") or [])
    if p is not None:
        return p
    try:
        lp = (((sub_data or {}).get("billing_info") or {}).get("last_payment") or {})
        return _parse_price((lp.get("amount") or {}).get("value"))
    except Exception:
        return None


async def _fetch_plan_list_price(plan_id, tier, access_token, paypal_api_base) -> Optional[float]:
    """Precio de lista AUTORITATIVO del plan. Primero GET del plan en PayPal
    (merchant-side, inmune a tampering del cliente); fallback a env
    PAYPAL_PLAN_{TIER}[_ANNUAL]_PRICE. None si no se puede determinar → fail-cheap."""
    if plan_id and access_token:
        try:
            async with httpx.AsyncClient(timeout=_HTTPX_TIMEOUT_S) as client:
                r = await client.get(
                    f"{paypal_api_base}/v1/billing/plans/{plan_id}",
                    headers={"Authorization": f"Bearer {access_token}"},
                )
                if r.status_code == 200:
                    p = _extract_regular_fixed_price(r.json().get("billing_cycles") or [])
                    if p is not None:
                        return p
        except Exception as e:
            logger.warning(f"[P1-BILLING-AMOUNT] GET plan {plan_id} falló: {type(e).__name__}")
    # [P1-BILLING-AMOUNT-FP-FIX · 2026-07-12] Fallback env por CADENCIA correcta. El
    # plan_id distingue mensual vs anual (el mapping ya conoce el sufijo _ANNUAL); usar
    # el env que corresponde en vez de "mensual primero" (que daba un piso equivocado:
    # under-detection para anuales, falso-positivo para mensuales si solo estaba el env
    # anual). Si no se puede determinar la cadencia, se prueban ambos (mejor un piso
    # aproximado que None) pero se emite WARN — el fallback debe ser observable.
    tier_u = (tier or "").upper()
    annual_plan_id = os.environ.get(f"PAYPAL_PLAN_{tier_u}_ANNUAL_ID")
    monthly_plan_id = os.environ.get(f"PAYPAL_PLAN_{tier_u}_ID")
    if plan_id and annual_plan_id and plan_id == annual_plan_id:
        suffixes = ("_ANNUAL",)
    elif plan_id and monthly_plan_id and plan_id == monthly_plan_id:
        suffixes = ("",)
    else:
        suffixes = ("", "_ANNUAL")  # cadencia indeterminada → ambos (aprox)
    for suffix in suffixes:
        v = _parse_price(os.environ.get(f"PAYPAL_PLAN_{tier_u}{suffix}_PRICE"))
        if v is not None:
            logger.warning(
                f"[P1-BILLING-AMOUNT] GET plan falló; usando precio-lista de env "
                f"PAYPAL_PLAN_{tier_u}{suffix}_PRICE={v} (cadencia={'ambigua' if len(suffixes) > 1 else suffix or 'mensual'})."
            )
            return v
    return None


async def _validate_discount_code(code: str, tier: str) -> Optional[dict]:
    """Re-valida un código de descuento server-side (MISMAS reglas que
    /api/discount/validate: is_active, vigencia, max_uses, applicable_tiers).
    Retorna {'discount_percent': N} si es válido para el tier, o None. SSOT para el
    lado de INTEGRIDAD (P1-BILLING-AMOUNT); el endpoint conserva sus mensajes de UX."""
    code = (code or "").strip().upper()
    tier = (tier or "").strip().lower()
    if not code:
        return None
    rows = await _run_sync_db_in_thread(
        lambda: execute_sql_query(
            """
            SELECT discount_percent, max_uses, current_uses, applicable_tiers,
                   valid_from::text AS valid_from, valid_until::text AS valid_until
              FROM public.discount_codes
             WHERE code = %s AND is_active = TRUE
            """,
            (code,),
            fetch_all=True,
        )
    )
    if not rows:
        return None
    d = rows[0]
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    try:
        if d.get("valid_from") and now < datetime.fromisoformat(d["valid_from"].replace("Z", "+00:00")):
            return None
        if d.get("valid_until") and now > datetime.fromisoformat(d["valid_until"].replace("Z", "+00:00")):
            return None
    except (ValueError, AttributeError):
        return None
    if d.get("max_uses") is not None and d.get("current_uses", 0) >= d["max_uses"]:
        return None
    applicable = d.get("applicable_tiers") or []
    if applicable and tier and tier not in applicable:
        return None
    return {"discount_percent": d.get("discount_percent")}


def _redeem_discount_code(code: str) -> None:
    """Hardening: incrementa current_uses del cupón al otorgar el tier. Best-effort
    (no fatal): sin esto un cupón con max_uses nunca se marcaba consumido (reuse
    ilimitado). Atómico vía WHERE con guard de max_uses."""
    code = (code or "").strip().upper()
    if not code:
        return
    try:
        execute_sql_write(
            """
            UPDATE public.discount_codes
               SET current_uses = COALESCE(current_uses, 0) + 1
             WHERE code = %s AND is_active = TRUE
               AND (max_uses IS NULL OR COALESCE(current_uses, 0) < max_uses)
            """,
            (code,),
        )
    except Exception as e:
        logger.warning(f"[P1-BILLING-AMOUNT] No se pudo redimir cupón {code}: {type(e).__name__}")


#: [P1-PLAN-LOTE-61 · 2026-09-15] Un cupón que venció hace menos de esto todavía «existe» para decidir si un override
#: sin cupón válido PUEDE ser legítimo: el caso del fail-cheap es el cupón que caducó entre crear la suscripción y /verify.
_CUPON_GRACIA_H = 24


async def _active_coupon_exists_for_tier(tier: str) -> Optional[bool]:
    """[P1-PLAN-LOTE-61 · 2026-09-15] (G1) ¿Existe ALGÚN cupón activo que pudiera aplicar a este tier? `True`/`False`, o
    `None` si no se pudo consultar (fail-cheap: sin respuesta no se bloquea nada).

    Las reglas son las de `_validate_discount_code` (`is_active`, vigencia, `applicable_tiers`) con dos holguras a
    propósito, porque aquí la pregunta no es «¿vale ESTE cupón?» sino «¿podría haber un cupón detrás de este override?»:
    un cupón agotado (`max_uses`) sigue contando, y uno vencido hace menos de `_CUPON_GRACIA_H` horas también. La
    auditoría del 22-ago lo midió: `discount_codes` estaba VACÍA, así que cualquier override era ilegítimo por
    definición — y el acceso se concedía igual. tooltip-anchor: P1-PLAN-LOTE-61-SIN-CUPON"""
    tier = (tier or "").strip().lower()
    try:
        rows = await _run_sync_db_in_thread(
            lambda: execute_sql_query(
                """
                SELECT applicable_tiers, valid_from::text AS valid_from, valid_until::text AS valid_until
                  FROM public.discount_codes
                 WHERE is_active = TRUE
                """,
                None,
                fetch_all=True,
            )
        )
    except Exception as e:
        logger.warning(f"[P1-PLAN-LOTE-61] no se pudo consultar discount_codes ({type(e).__name__}): no se bloquea")
        return None
    if rows is None:
        return None
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone.utc)
    for d in rows:
        try:
            if d.get("valid_from") and now < datetime.fromisoformat(str(d["valid_from"]).replace("Z", "+00:00")):
                continue
            if d.get("valid_until") and now - timedelta(hours=_CUPON_GRACIA_H) > datetime.fromisoformat(
                    str(d["valid_until"]).replace("Z", "+00:00")):
                continue
        except (ValueError, AttributeError, TypeError):
            return True          # una fila ilegible podría ser un cupón: no se asume que no existe
        applicable = d.get("applicable_tiers") or []
        if applicable and tier and tier not in applicable:
            continue
        return True
    return False


async def _verify_subscription_amount(
    *, sub_data, verified_plan_id, tier, coupon_code, access_token, paypal_api_base,
    user_id, subscription_id,
) -> None:
    """Verifica que el monto pactado en la suscripción NO fue manipulado por debajo
    del precio de lista (menos un descuento re-validado server-side). Alerta siempre
    ante sospecha; en modo 'block' 409ea SOLO con underpayment PROBADO (fail-cheap).

    [P1-PLAN-LOTE-61 · 2026-09-15] (G1, decisión del dueño delegada el 14-sep: `block`) El default pasa de `warn` a
    `block`, y «probado» incluye ahora el override por debajo de la lista cuando NO existe ningún cupón activo aplicable
    al tier (`_active_coupon_exists_for_tier` → `False`): ningún cupón puede justificarlo. Si existe alguno (el del
    usuario no llegó o caducó) o no se pudo consultar, sigue siendo ambiguo: alerta y no bloquea. Rollback sin
    redeploy: `MEALFIT_BILLING_VERIFY_AMOUNT=warn`."""
    mode = _env_str("MEALFIT_BILLING_VERIFY_AMOUNT", "block", {"off", "warn", "block"}).lower()
    if mode == "off" or not sub_data:
        return
    # Sin override → precio estándar del plan, sin manipulación posible.
    if not bool(sub_data.get("plan_overridden")):
        return

    disc = await _validate_discount_code(coupon_code, tier)
    discount_pct = disc.get("discount_percent") if disc else None

    actual_price = _extract_override_price(sub_data)
    list_price = await _fetch_plan_list_price(verified_plan_id, tier, access_token, paypal_api_base)
    tol = _env_float("MEALFIT_BILLING_AMOUNT_TOLERANCE_PCT", 0.02)

    expected_min = None
    if list_price is not None:
        expected_min = list_price * (1.0 - (float(discount_pct or 0) / 100.0))

    # [P1-BILLING-AMOUNT-FP-FIX · 2026-07-12] `proven_underpaid` EXIGE un cupón
    # re-validado (discount_pct is not None): solo es PRUEBA de fraude un precio por
    # debajo del piso de un cupón que SÍ existe server-side. Un override SIN cupón
    # válido re-validado es AMBIGUO — puede ser un atacante ($0.01 sin cupón) pero
    # también un pago legítimo cuyo coupon_code no llegó a /verify (frontend viejo
    # cacheado por PWA, o cupón que se agotó/expiró entre la creación de la sub y el
    # verify). Bloquear ese caso ambiguo rechazaría pagos legítimos ya cobrados por
    # PayPal → limbo. Por eso el caso ambiguo es warn+alerta (nunca block), coherente
    # con el comentario de abajo. (Review adversario 2026-07-12: el código previo
    # bloqueaba el caso ambiguo, contradiciendo su propio comentario.)
    proven_underpaid = (
        discount_pct is not None
        and actual_price is not None
        and expected_min is not None
        and actual_price < expected_min * (1.0 - tol)
    )
    # Sospecha (→ alerta): underpayment probado bajo el cupón, O un override SIN cupón
    # válido que lo justifique (ambiguo — se alerta pero NO se bloquea).
    suspicious = proven_underpaid or (discount_pct is None)
    if not suspicious:
        return  # cupón válido + monto coherente

    # [P1-PLAN-LOTE-61] Sin cupón válido y por debajo de la lista: ¿existe ALGÚN cupón que pudiera explicarlo? Si no
    # existe ninguno, el override es ilegítimo por definición y el ambiguo deja de serlo.
    sin_cupon_posible = False
    if (discount_pct is None and actual_price is not None and list_price is not None
            and actual_price < list_price * (1.0 - tol)):
        sin_cupon_posible = (await _active_coupon_exists_for_tier(tier)) is False

    _persist_billing_alert(
        alert_key=f"billing_price_tampering:{user_id}:{subscription_id}",
        severity="critical",
        title="Verify: posible manipulación del monto de la suscripción",
        message=(
            f"User {user_id}: sub {subscription_id} (plan {verified_plan_id}, tier {tier}) "
            f"tiene plan_overridden con precio={actual_price} vs esperado_min={expected_min} "
            f"(lista={list_price}, cupón={coupon_code or '∅'} → {discount_pct}%). "
            f"proven_underpaid={proven_underpaid}, sin_cupon_posible={sin_cupon_posible}, modo={mode}."
        ),
        metadata={
            "user_id": user_id, "sub_id": subscription_id, "tier": tier,
            "actual_price": actual_price, "list_price": list_price,
            "expected_min": expected_min, "coupon": coupon_code or None,
            "discount_pct": discount_pct, "proven_underpaid": proven_underpaid,
            "sin_cupon_posible": sin_cupon_posible,
        },
    )
    logger.warning(
        f"⚠️ [P1-BILLING-AMOUNT] Sospecha de manipulación de monto: user={user_id} "
        f"sub={subscription_id} actual={actual_price} expected_min={expected_min} "
        f"proven_underpaid={proven_underpaid} sin_cupon_posible={sin_cupon_posible} modo={mode}"
    )
    # Bloquear SOLO con underpayment PROBADO bajo un cupón re-validado (precio <
    # piso del cupón). El caso ambiguo (sin cupón válido / no-parseable / cupón no
    # reenviado) queda en warn+alerta → fail-cheap, nunca bloquea un pago que no
    # podemos PROBAR fraudulento.
    if mode == "block" and (proven_underpaid or sin_cupon_posible):
        raise HTTPException(
            status_code=409,
            detail="El monto de la suscripción no coincide con el precio del plan. Contacta soporte.",
        )


@router.get("/status")
async def api_subscription_status(
    verified_user_id: Optional[str] = Depends(get_verified_user_id),
    _rl: None = Depends(_SUBSCRIPTION_STATUS_LIMITER),
):
    """[P1-PLAN-LOTE-714 · 2026-09-28] Lo que la tarjeta «Suscripción» de Configuración necesita, en una lectura.

    Respuesta (200), siempre las mismas seis claves:
        {
          "has_paypal_subscription": bool,         # el perfil tiene paypal_subscription_id
          "subscription_status":     str | null,   # tal cual en el perfil (ACTIVE, CANCELLED, PAYMENT_RETRYING…)
          "paypal_status":           str | null,   # estado VIVO en PayPal; null si no hay sub o PayPal no respondió
          "next_billing_time":       ISO | null,   # próximo cobro según PayPal; solo con la sub ACTIVE y sin cancelar
          "access_until":            ISO | null,   # solo si el perfil está CANCELLED: la fecha que se honra
          "plan_tier":               str | null    # tal cual en el perfil
        }

    Cero IA: `get_verified_user_id` + RateLimiter, sin paywall. Solo lee la fila del id del JWT. Si PayPal falla o no
    está configurado, fail-soft: los campos del perfil y `paypal_status: null` (es una lectura; queda un warning). Nunca
    devuelve el payload de PayPal (email, dirección del comprador) ni tokens.
    tooltip-anchor: P1-PLAN-LOTE-714-STATUS"""
    if not verified_user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    try:
        fila = await _run_sync_db_in_thread(
            lambda: execute_sql_query(
                "SELECT plan_tier, subscription_status, paypal_subscription_id, subscription_end_date "
                "  FROM public.user_profiles WHERE id = %s",
                (verified_user_id,),
                fetch_one=True,
            )
        )
    except Exception as e:
        logger.error(f"❌ [P1-PLAN-LOTE-714] /status: no se pudo leer el perfil: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=safe_error_detail(e))
    fila = fila or {}
    sub_id = fila.get("paypal_subscription_id") or None
    estado = fila.get("subscription_status") or None
    cancelada = estado == "CANCELLED"
    respuesta = {
        "has_paypal_subscription": bool(sub_id),
        "subscription_status": estado,
        "paypal_status": None,
        "next_billing_time": None,
        "access_until": _as_utc_iso(fila.get("subscription_end_date")) if cancelada else None,
        "plan_tier": fila.get("plan_tier"),
    }
    if sub_id:
        snapshot = await _paypal_status_snapshot(str(sub_id))
        if snapshot:
            respuesta["paypal_status"] = snapshot.get("status")
            # Un próximo cobro solo existe si la sub sigue viva en los DOS lados: con el perfil CANCELLED, un
            # «ACTIVE» cacheado de hace unos minutos no puede anunciar un cobro que ya no va a ocurrir.
            if not cancelada and snapshot.get("status") == "ACTIVE":
                respuesta["next_billing_time"] = snapshot.get("next_billing_time")
    return respuesta


@router.post("/verify")
async def api_verify_subscription(data: dict = Body(...), verified_user_id: str = Depends(get_verified_user_id)):
    try:
        user_id = data.get("user_id")
        subscription_id = data.get("subscriptionID")
        client_hint_tier = (data.get("tier") or "").strip().lower()
        # [P1-BILLING-AMOUNT · 2026-07-12] Cupón aplicado (opcional; retrocompat con
        # frontend viejo que aún no lo reenvía). Se RE-VALIDA server-side — el
        # discount_percent del cliente NO se confía.
        coupon_code = (data.get("coupon_code") or "").strip().upper()

        if not user_id or not subscription_id or not client_hint_tier:
            raise HTTPException(status_code=400, detail="Missing parameters")

        if not verified_user_id or verified_user_id != user_id:
            logger.warning(f"Intento no autorizado de verificar suscripcion: req_user={user_id}, auth_user={verified_user_id}")
            raise HTTPException(status_code=401, detail="No autorizado.")

        PAYPAL_CLIENT_ID = os.environ.get("PAYPAL_CLIENT_ID")
        PAYPAL_SECRET = os.environ.get("PAYPAL_SECRET")
        is_sandbox = not is_production()  # [P2-PROD-AUDIT-3] SSOT normalizado (lower+strip)
        PAYPAL_API_BASE = "https://api-m.sandbox.paypal.com" if is_sandbox else "https://api-m.paypal.com"

        # [P0-BILLING-2 · 2026-05-12] Fail-secure si faltan env vars PayPal
        # en producción. Pre-fix: cuando ambas keys eran falsy, el handler
        # forzaba `success` a True (bypass completo de verificación). Si el
        # contenedor perdía las env vars (rotación rota, misconfig del VPS Oracle),
        # CUALQUIER POST con `subscription_id` arbitrario upgradeaba el plan.
        # Anchor: P0-BILLING-2-FAIL-SECURE.
        env_ready = bool(PAYPAL_CLIENT_ID and PAYPAL_SECRET)
        allow_bypass = (
            os.environ.get("MEALFIT_ALLOW_PAYPAL_BYPASS", "").lower()
            in ("1", "true", "yes")
        )
        if not env_ready and not is_sandbox and not allow_bypass:
            logger.error(
                "❌ [P0-BILLING-2] PAYPAL_CLIENT_ID/PAYPAL_SECRET ausentes en "
                "producción. Configurar env vars o (SOLO dev) setear "
                "MEALFIT_ALLOW_PAYPAL_BYPASS=true."
            )
            raise HTTPException(
                status_code=503,
                detail="Payment provider misconfigured. Contact support.",
            )

        access_token = None
        verified_plan_id = None
        sub_data = None  # [P1-BILLING-AMOUNT] capturado para la verificación de monto

        if env_ready and PAYPAL_CLIENT_ID and PAYPAL_SECRET:
            async with httpx.AsyncClient(timeout=_HTTPX_TIMEOUT_S) as client:
                auth_resp = await client.post(
                    f"{PAYPAL_API_BASE}/v1/oauth2/token",
                    auth=(PAYPAL_CLIENT_ID, PAYPAL_SECRET),
                    data={"grant_type": "client_credentials"}
                )
                if auth_resp.status_code != 200:
                    logger.error(f"Error auth Paypal: {auth_resp.text}")
                    raise HTTPException(status_code=500, detail="Error de autenticación con proveedor de pagos.")

                access_token = auth_resp.json().get("access_token")

                sub_resp = await client.get(
                    f"{PAYPAL_API_BASE}/v1/billing/subscriptions/{subscription_id}",
                    headers={"Authorization": f"Bearer {access_token}"}
                )
                if sub_resp.status_code != 200:
                    logger.error(f"Error fetching sub: {sub_resp.text}")
                    raise HTTPException(status_code=400, detail="La suscripción no fue reconocida por PayPal.")

                sub_data = sub_resp.json()
                status = sub_data.get("status")

                if status != "ACTIVE":
                    raise HTTPException(status_code=400, detail=f"Suscripción no válida. Estado actual: {status}")

                verified_plan_id = sub_data.get("plan_id")
        else:
            logger.warning(
                "⚠️ [DEV-BYPASS] PayPal env vars ausentes; bypass permitido por "
                "sandbox/MEALFIT_ALLOW_PAYPAL_BYPASS. NO en producción."
            )

        # [P0-BILLING-1] Derivar tier server-side desde verified_plan_id, NO
        # desde data.get("tier"). El client_hint_tier solo se usa para log de
        # divergencia (no afecta el UPDATE).
        if verified_plan_id:
            plan_tier_map = _build_paypal_plan_tier_map()
            server_tier = plan_tier_map.get(verified_plan_id)
            if not server_tier:
                logger.warning(
                    f"[P0-BILLING-1] PayPal plan_id={verified_plan_id!r} no "
                    f"mapea a tier interno. Configurar env vars mensuales "
                    f"PAYPAL_PLAN_BASIC_ID/PAYPAL_PLAN_PLUS_ID/"
                    f"PAYPAL_PLAN_ULTRA_ID y anuales (P1-BILLING-ANNUAL-MAP) "
                    f"PAYPAL_PLAN_BASIC_ANNUAL_ID/PAYPAL_PLAN_PLUS_ANNUAL_ID/"
                    f"PAYPAL_PLAN_ULTRA_ANNUAL_ID. cliente hint: {client_hint_tier!r}."
                )
                raise HTTPException(
                    status_code=400,
                    detail="Plan no reconocido. Contacta soporte.",
                )
            if client_hint_tier and client_hint_tier != server_tier:
                logger.warning(
                    f"[P0-BILLING-1] tier mismatch: cliente={client_hint_tier!r} "
                    f"vs server={server_tier!r} (plan_id={verified_plan_id}). "
                    f"Asignando server-side."
                )
            tier_to_assign = server_tier
        elif allow_bypass or is_sandbox:
            if client_hint_tier not in ("basic", "plus", "ultra"):
                raise HTTPException(status_code=400, detail="tier inválido.")
            tier_to_assign = client_hint_tier
        else:
            # Defensa-en-profundidad: el gate de arriba ya levantó 503 si
            # env_ready=False AND not sandbox AND not bypass. Si llegamos
            # acá hay un bug — fallar antes de mutar.
            raise HTTPException(status_code=500, detail="No se pudo derivar tier.")

        logger.info(
            f"✅ Subscripcion Verificada B2B ({subscription_id}) para usuario "
            f"{user_id}. Tier asignado server-side: {tier_to_assign}"
        )

        # [P1-BILLING-AMOUNT · 2026-07-12] Verificar integridad del MONTO ANTES de
        # cancelar la sub vieja / aplicar el tier: si es tampering en modo 'block',
        # 409ea sin haber tocado nada. En 'warn' (default) solo alerta y continúa.
        await _verify_subscription_amount(
            sub_data=sub_data,
            verified_plan_id=verified_plan_id,
            tier=tier_to_assign,
            coupon_code=coupon_code,
            access_token=access_token,
            paypal_api_base=PAYPAL_API_BASE,
            user_id=user_id,
            subscription_id=subscription_id,
        )

        existing_row = await _run_sync_db_in_thread(
            lambda: execute_sql_query(
                "SELECT paypal_subscription_id, subscription_status FROM public.user_profiles WHERE id = %s",
                (user_id,),
                fetch_one=True,
            )
        )
        if existing_row:
            old_sub_id = existing_row.get("paypal_subscription_id")
            old_status = existing_row.get("subscription_status")

            if old_sub_id and old_sub_id != subscription_id and old_status != "INACTIVE":
                logger.info(f"🔄 Detectado Upgrade/Cambio. Cancelando suscripción antigua {old_sub_id} en PayPal...")
                if access_token:
                    async with httpx.AsyncClient(timeout=_HTTPX_TIMEOUT_S) as cancel_client:
                        cancel_resp = await cancel_client.post(
                            f"{PAYPAL_API_BASE}/v1/billing/subscriptions/{old_sub_id}/cancel",
                            headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"},
                            json={"reason": "Upgrade o cambio a una nueva suscripción."}
                        )
                        # [P1-BILLING-UPGRADE-FAIL-LOUD · 2026-05-12] Pre-fix
                        # esta rama solo emitía `logger.warning` cuando la
                        # cancel respondía != 204 y SEGUÍA al UPDATE de
                        # plan_tier. La sub vieja quedaba ACTIVE en PayPal
                        # cobrando recurrentemente, mientras la nueva también
                        # cobraba → cliente paga 2× hasta intervención manual.
                        # Ahora fail-loud: persistir alert + raise 409 antes
                        # del UPDATE. Trata 404 / 422-already-cancelled como
                        # éxito idempotente.
                        if _is_paypal_cancel_idempotent_success(cancel_resp.status_code, cancel_resp.text):
                            logger.info(
                                f"✅ Suscripción antigua {old_sub_id} cancelada "
                                f"(status={cancel_resp.status_code})."
                            )
                        else:
                            _persist_billing_alert(
                                alert_key=f"billing_old_sub_cancel_failed:{user_id}:{old_sub_id}",
                                severity="critical",
                                title="Upgrade: cancel de sub vieja falló — riesgo doble cobro",
                                message=(
                                    f"User {user_id} upgrade de sub {old_sub_id} a "
                                    f"{subscription_id}. PayPal /cancel respondió "
                                    f"status={cancel_resp.status_code}: "
                                    f"{(cancel_resp.text or '')[:300]}. UPDATE de "
                                    f"plan_tier ABORTADO para evitar estado divergente "
                                    f"(BD nueva sub + PayPal cobrando ambas)."
                                ),
                                metadata={
                                    "user_id": user_id,
                                    "old_sub_id": old_sub_id,
                                    "new_sub_id": subscription_id,
                                    "paypal_status": cancel_resp.status_code,
                                },
                            )
                            logger.error(
                                f"❌ [P1-BILLING-UPGRADE-FAIL-LOUD] Cancel de sub "
                                f"vieja {old_sub_id} falló (status="
                                f"{cancel_resp.status_code}): "
                                f"{(cancel_resp.text or '')[:200]}. Abortando "
                                f"upgrade — el cliente debe reintentar para "
                                f"evitar doble cobro."
                            )
                            raise HTTPException(
                                status_code=409,
                                detail=(
                                    "No se pudo cancelar la suscripción anterior "
                                    "en PayPal. Por favor reintenta en unos "
                                    "minutos; si persiste, contacta soporte."
                                ),
                            )

        # [P1-NEON-DB-MIGRATION · 2026-06-12] El payload conserva la forma
        # dict para que el anchor P0-BILLING-1 quede explícito: `plan_tier`
        # viene de `tier_to_assign` (server-derived, invariante I-Billing-1)
        # — NUNCA del body del cliente. `RETURNING id` preserva el check
        # fail-loud de filas matcheadas de abajo.
        # [P1-PLAN-LOTE-714 · 2026-09-28] `subscription_end_date = NULL`: la fecha de fin la escribe una cancelación y
        # nadie la borraba al volver a suscribirse, así que Configuración pintaba como «Próximo cobro» (o «Acceso
        # hasta») la fecha de la suscripción ANTERIOR. Una sub recién activada no tiene fin.
        # tooltip-anchor: P1-PLAN-LOTE-714-END-DATE-CLEAR
        update_payload = {
            "plan_tier": tier_to_assign,
            "paypal_subscription_id": subscription_id,
            "subscription_status": "ACTIVE",
        }
        updated_rows = await _run_sync_db_in_thread(
            lambda: execute_sql_write(
                """
                UPDATE public.user_profiles
                   SET plan_tier = %s,
                       paypal_subscription_id = %s,
                       subscription_status = %s,
                       subscription_end_date = NULL
                 WHERE id = %s
                RETURNING id
                """,
                (
                    update_payload["plan_tier"],
                    update_payload["paypal_subscription_id"],
                    update_payload["subscription_status"],
                    user_id,
                ),
                returning=True,
            )
        )

        # [P1-PROD-AUDIT-3 · 2026-05-30] Verificar que el UPDATE tocó una fila.
        # PayPal YA validó/cobró en este punto; si `WHERE id = user_id` matcheó 0
        # filas (perfil ausente — caso patológico, el trigger handle_new_user lo
        # crea al signup) el upgrade se perdía en silencio con success:true. Fail-
        # loud + alert para reconciliación manual, espejo del cancel path arriba.
        if not updated_rows:
            _persist_billing_alert(
                alert_key=f"billing_profile_not_found_on_upgrade:{user_id}:{subscription_id}",
                severity="critical",
                title="Verify: perfil no encontrado al aplicar tier",
                message=(
                    f"User {user_id}: PayPal validó sub {subscription_id} (tier "
                    f"{tier_to_assign}) pero el UPDATE de user_profiles matcheó 0 "
                    f"filas. El usuario pagó pero el tier no se aplicó — reconciliar."
                ),
                metadata={"user_id": user_id, "sub_id": subscription_id, "tier": tier_to_assign},
            )
            raise HTTPException(
                status_code=500,
                detail="Perfil no encontrado al aplicar el plan. Contacta soporte; tu pago está registrado.",
            )

        # [P1-BILLING-AMOUNT · 2026-07-12] Hardening: marcar el cupón como consumido
        # (best-effort) SOLO si re-valida (no quemar usos por un código inválido).
        # Sin esto un cupón con max_uses nunca se marcaba usado → reuse ilimitado.
        # [P1-BILLING-AMOUNT-FP-FIX · 2026-07-12] Idempotencia por subscription_id: si
        # `existing_row` (leído ANTES del UPDATE) ya tenía ESTA sub, es un reintento/
        # doble-submit del mismo /verify → NO re-redimir (evita inflar current_uses por
        # una única redención real). Un reintento fetcha el perfil ya actualizado por
        # la primera llamada, así que su paypal_subscription_id == subscription_id.
        _sub_already_applied = bool(
            existing_row and existing_row.get("paypal_subscription_id") == subscription_id
        )
        if coupon_code and not _sub_already_applied:
            try:
                if await _validate_discount_code(coupon_code, tier_to_assign):
                    await _run_sync_db_in_thread(lambda: _redeem_discount_code(coupon_code))
            except Exception as _e:
                logger.warning(f"[P1-BILLING-AMOUNT] redención de cupón no fatal falló: {type(_e).__name__}")

        # [P1-PLAN-LOTE-714] Lo que `/status` tuviera cacheado de la sub nueva o de la vieja ya no vale.
        _invalidate_paypal_status_cache(
            subscription_id, existing_row.get("paypal_subscription_id") if existing_row else None
        )
        return {"success": True, "message": "Suscripción verificada y plan actualizado B2B."}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error interno en /api/subscription/verify: {str(e)}")
        raise HTTPException(status_code=500, detail=safe_error_detail(e))

# ---------------------------------------------------------------------------
# [P1-ACCOUNT-DELETE-1 · 2026-06-22] Helper SSOT reutilizable para cancelar una
# suscripción PayPal, con la MISMA política fail-loud que `/cancel`. Lo consume
# el flujo de eliminación de cuenta (`POST /api/account/delete`): hay que cancelar
# en PayPal ANTES de borrar `user_profiles` (que guarda `paypal_subscription_id`),
# o PayPal seguiría cobrando indefinidamente sin handle para detenerlo.
#
# NO toca la BD (el caller decide). Retorna `end_date` ISO en éxito/idempotente;
# `raise HTTPException(502/503)` + alerta en fallo real → el caller ABORTA el
# borrado (nunca dejamos una sub viva tras "eliminar cuenta").
# ---------------------------------------------------------------------------
async def cancel_paypal_subscription_for_user(
    user_id: str, subscription_id: str, *, reason: str
) -> Optional[str]:
    PAYPAL_CLIENT_ID = os.environ.get("PAYPAL_CLIENT_ID")
    PAYPAL_SECRET = os.environ.get("PAYPAL_SECRET")
    is_sandbox = not is_production()
    PAYPAL_API_BASE = "https://api-m.sandbox.paypal.com" if is_sandbox else "https://api-m.paypal.com"
    env_ready = bool(PAYPAL_CLIENT_ID and PAYPAL_SECRET)
    allow_bypass = (
        os.environ.get("MEALFIT_ALLOW_PAYPAL_BYPASS", "").lower() in ("1", "true", "yes")
    )
    # Fail-secure: sin credenciales en producción NO seguimos — borrar la cuenta
    # perdería el subscription_id mientras PayPal sigue cobrando (P0-BILLING-2).
    if not env_ready and not is_sandbox and not allow_bypass:
        logger.error(
            "❌ [P1-ACCOUNT-DELETE-1] PAYPAL_CLIENT_ID/PAYPAL_SECRET ausentes en "
            "producción — cancel pre-borrado rechazado para no perder el handle "
            "de una sub que seguiría cobrando."
        )
        raise HTTPException(status_code=503, detail="Payment provider misconfigured. Contact support.")
    if not env_ready:
        # sandbox/bypass dev sin credenciales reales: no hay sub que cancelar de verdad.
        logger.warning("[P1-ACCOUNT-DELETE-1] PayPal env ausente (sandbox/bypass) — cancel omitido.")
        return None

    end_date = None
    # [P1-PLAN-LOTE-714 · 2026-09-28] Un timeout o un corte de red con PayPal (excepciones de httpx) salía como 500
    # genérico y SIN alerta: el mismo hueco que tenía `/cancel` (I-Billing-3). El borrado se aborta igual, pero ahora
    # queda la alerta crítica y el usuario recibe un 502 que dice qué pasó.
    try:
        async with httpx.AsyncClient(timeout=_HTTPX_TIMEOUT_S) as client:
            auth_resp = await client.post(
                f"{PAYPAL_API_BASE}/v1/oauth2/token",
                auth=(PAYPAL_CLIENT_ID, PAYPAL_SECRET),
                data={"grant_type": "client_credentials"},
            )
            if auth_resp.status_code != 200:
                _persist_billing_alert(
                    alert_key=f"billing_cancel_failed:{user_id}:{subscription_id}",
                    severity="critical",
                    title="Account-delete: OAuth PayPal falló — cancel no notificado",
                    message=(
                        f"User {user_id} eliminó su cuenta; cancel de sub {subscription_id} "
                        f"no se pudo notificar (OAuth status={auth_resp.status_code}: "
                        f"{(auth_resp.text or '')[:300]}). Borrado ABORTADO."
                    ),
                    metadata={"user_id": user_id, "sub_id": subscription_id, "stage": "oauth", "flow": "account_delete"},
                )
                raise HTTPException(
                    status_code=502,
                    detail="No se pudo contactar el proveedor de pagos. Reintenta en unos minutos; si persiste, contacta soporte.",
                )
            access_token = auth_resp.json().get("access_token")

            sub_info_resp = await client.get(
                f"{PAYPAL_API_BASE}/v1/billing/subscriptions/{subscription_id}",
                headers={"Authorization": f"Bearer {access_token}"},
            )
            if sub_info_resp.status_code == 200:
                end_date = sub_info_resp.json().get("billing_info", {}).get("next_billing_time")

            cancel_resp = await client.post(
                f"{PAYPAL_API_BASE}/v1/billing/subscriptions/{subscription_id}/cancel",
                headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"},
                json={"reason": reason},
            )
            if not _is_paypal_cancel_idempotent_success(cancel_resp.status_code, cancel_resp.text):
                _persist_billing_alert(
                    alert_key=f"billing_cancel_failed:{user_id}:{subscription_id}",
                    severity="critical",
                    title="Account-delete: PayPal rechazó la cancelación — riesgo cobro post-borrado",
                    message=(
                        f"User {user_id} eliminó su cuenta; PayPal /cancel de sub {subscription_id} "
                        f"respondió status={cancel_resp.status_code}: {(cancel_resp.text or '')[:300]}. "
                        f"Borrado ABORTADO para no perder el handle de una sub viva."
                    ),
                    metadata={"user_id": user_id, "sub_id": subscription_id, "stage": "cancel", "flow": "account_delete"},
                )
                raise HTTPException(
                    status_code=502,
                    detail="El proveedor de pagos rechazó la cancelación. Reintenta en unos minutos; si persiste, contacta soporte.",
                )
    except httpx.HTTPError as _e_red:
        _persist_billing_alert(
            alert_key=f"billing_cancel_failed:{user_id}:{subscription_id}",
            severity="critical",
            title="Account-delete: PayPal no respondió — cancel no notificado",
            message=(
                f"User {user_id} eliminó su cuenta; la cancelación de la sub {subscription_id} no llegó a PayPal "
                f"({type(_e_red).__name__}). Borrado ABORTADO para no perder el handle de una sub viva."
            ),
            metadata={"user_id": user_id, "sub_id": subscription_id, "stage": "network", "flow": "account_delete",
                      "error": type(_e_red).__name__},
        )
        raise HTTPException(
            status_code=502,
            detail="No se pudo contactar el proveedor de pagos. Reintenta en unos minutos; si persiste, contacta soporte.",
        )
    logger.info(f"✅ [P1-ACCOUNT-DELETE-1] Sub {subscription_id} de {user_id} cancelada antes de eliminar la cuenta.")
    return end_date


@router.post("/cancel")
async def api_cancel_subscription(
    data: Optional[dict] = Body(None),
    verified_user_id: Optional[str] = Depends(get_verified_user_id),
    _rl: None = Depends(_SUBSCRIPTION_CANCEL_LIMITER),
):
    """Cancela la suscripción PayPal del usuario del JWT.

    [P1-PLAN-LOTE-714 · 2026-09-28] Respuesta (200):
        {"success": true, "message": str, "access_until": ISO-8601 | null}
    `access_until` es la fecha que el cargador del perfil honrará antes de degradar; null solo si no se pudo
    determinar (entonces el plan se degrada ya, como antes, y queda una alerta para el operador). Errores: 401 (sin
    sesión, o un `user_id` ajeno en el body), 400 (sin suscripción), 429 (limitador), 502 (PayPal rechazó o no
    respondió: la BD NO se toca), 503 (PayPal sin configurar en producción).
    tooltip-anchor: P1-PLAN-LOTE-714-CANCEL"""
    try:
        # [P1-PLAN-LOTE-714 · 2026-09-28] Quien cancela ES el usuario del JWT. El body era OBLIGATORIO y su `user_id`
        # solo servía para compararlo con el JWT: una llamada sin body — o con `Content-Type: application/json` y el
        # cuerpo vacío — moría en 422 antes de llegar aquí. Ahora el body es opcional; si trae `user_id` tiene que ser
        # el mismo del JWT (401 como siempre): nadie cancela «en nombre» de otro.
        # tooltip-anchor: P1-PLAN-LOTE-714-CANCEL-BODY
        body_user_id = data.get("user_id") if isinstance(data, dict) else None
        if not verified_user_id or (body_user_id is not None and body_user_id != verified_user_id):
            logger.warning(
                f"Intento no autorizado de cancelar suscripcion: req_user={body_user_id}, auth_user={verified_user_id}"
            )
            raise HTTPException(status_code=401, detail="No autorizado.")
        user_id = verified_user_id

        profile_row = await _run_sync_db_in_thread(
            lambda: execute_sql_query(
                "SELECT paypal_subscription_id, subscription_status, subscription_end_date "
                "  FROM public.user_profiles WHERE id = %s",
                (user_id,),
                fetch_one=True,
            )
        )
        if not profile_row or not profile_row.get("paypal_subscription_id"):
            raise HTTPException(status_code=400, detail="No active subscription found to cancel.")

        subscription_id = profile_row["paypal_subscription_id"]

        # [P1-PLAN-LOTE-714] Cancelar dos veces es idempotente y barato: la fila ya CANCELLED con su fecha de fin la
        # escribió un cancel que PayPal confirmó (este endpoint o el webhook de PayPal), así que no hay nada que
        # repetir allí — y repetirlo recalcularía una fecha que ya está bien.
        _fin_guardado = _as_utc_iso(profile_row.get("subscription_end_date"))
        if profile_row.get("subscription_status") == "CANCELLED" and _fin_guardado:
            return {
                "success": True,
                "message": _CANCEL_OK_MESSAGE if _is_future(_fin_guardado) else _CANCEL_OK_SIN_FECHA_MESSAGE,
                "access_until": _fin_guardado,
            }

        PAYPAL_CLIENT_ID = os.environ.get("PAYPAL_CLIENT_ID")
        PAYPAL_SECRET = os.environ.get("PAYPAL_SECRET")
        is_sandbox = not is_production()  # [P2-PROD-AUDIT-3] SSOT normalizado (lower+strip)
        PAYPAL_API_BASE = "https://api-m.sandbox.paypal.com" if is_sandbox else "https://api-m.paypal.com"

        # [P0-BILLING-2 · 2026-05-12] Fail-secure en /cancel también: sin env
        # vars PayPal en producción el flujo legacy actualizaba BD a CANCELLED
        # pero NO notificaba PayPal → cobro recurrente sigue mientras BD dice
        # cancelado. Anchor: P0-BILLING-2-FAIL-SECURE.
        env_ready = bool(PAYPAL_CLIENT_ID and PAYPAL_SECRET)
        allow_bypass = (
            os.environ.get("MEALFIT_ALLOW_PAYPAL_BYPASS", "").lower()
            in ("1", "true", "yes")
        )
        if not env_ready and not is_sandbox and not allow_bypass:
            logger.error(
                "❌ [P0-BILLING-2] PAYPAL_CLIENT_ID/PAYPAL_SECRET ausentes en "
                "producción. /cancel rechazado para evitar estado divergente "
                "(BD CANCELLED + PayPal siguiendo cobrando)."
            )
            raise HTTPException(
                status_code=503,
                detail="Payment provider misconfigured. Contact support.",
            )

        sub_snapshot = None        # la suscripción según PayPal (para saber hasta cuándo pagó)
        paypal_consultado = False  # True solo si PayPal confirmó la cancelación

        if env_ready and PAYPAL_CLIENT_ID and PAYPAL_SECRET:
            _etapa = "oauth"
            # [P1-PLAN-LOTE-714 · 2026-09-28] Un timeout o un fallo de red con PayPal (excepciones de httpx) acababa en
            # un 500 genérico SIN alerta: la cancelación quedaba en el aire (¿llegó a PayPal o no?) y nadie se
            # enteraba. Ahora es un fallo más del proveedor, con el mismo contrato que los otros dos de esta ruta
            # (I-Billing-3): alerta `billing_cancel_failed` con la etapa y 502, sin tocar la BD. Si PayPal sí llegó a
            # cancelar, el reintento del usuario es idempotente y el webhook CANCELLED lo reflejará de todos modos.
            # tooltip-anchor: P1-PLAN-LOTE-714-CANCEL-TIMEOUT
            try:
                async with httpx.AsyncClient(timeout=_HTTPX_TIMEOUT_S) as client:
                    auth_resp = await client.post(
                        f"{PAYPAL_API_BASE}/v1/oauth2/token",
                        auth=(PAYPAL_CLIENT_ID, PAYPAL_SECRET),
                        data={"grant_type": "client_credentials"}
                    )
                    # [P1-BILLING-CANCEL-FAIL-LOUD · 2026-05-12] Pre-fix: si la
                    # OAuth a PayPal fallaba (auth_resp != 200), el código
                    # SALTABA todo el bloque y procedía al UPDATE BD=CANCELLED
                    # sin haber notificado a PayPal — mismo modo de fallo que
                    # el cancel_resp legacy. Ahora fail-loud también en auth.
                    if auth_resp.status_code != 200:
                        _persist_billing_alert(
                            alert_key=f"billing_cancel_failed:{user_id}:{subscription_id}",
                            severity="critical",
                            title="Cancel: OAuth PayPal falló — cancel no notificado",
                            message=(
                                f"User {user_id} solicitó cancel de sub "
                                f"{subscription_id}. PayPal OAuth /token respondió "
                                f"status={auth_resp.status_code}: "
                                f"{(auth_resp.text or '')[:300]}. UPDATE de "
                                f"subscription_status ABORTADO — PayPal nunca "
                                f"recibió la cancel request."
                            ),
                            metadata={
                                "user_id": user_id,
                                "sub_id": subscription_id,
                                "paypal_oauth_status": auth_resp.status_code,
                                "stage": "oauth",
                            },
                        )
                        logger.error(
                            f"❌ [P1-BILLING-CANCEL-FAIL-LOUD] OAuth PayPal falló "
                            f"para cancel de {subscription_id} (status="
                            f"{auth_resp.status_code}). Abortando — BD mantiene "
                            f"estado ACTIVE para reflejar realidad."
                        )
                        raise HTTPException(
                            status_code=502,
                            detail=(
                                "No se pudo contactar el proveedor de pagos. "
                                "Por favor reintenta en unos minutos; si "
                                "persiste, contacta soporte."
                            ),
                        )

                    access_token = auth_resp.json().get("access_token")

                    _etapa = "subscription_get"
                    sub_info_resp = await client.get(
                        f"{PAYPAL_API_BASE}/v1/billing/subscriptions/{subscription_id}",
                        headers={"Authorization": f"Bearer {access_token}"}
                    )
                    if sub_info_resp.status_code == 200:
                        sub_snapshot = _json_object_or_none(sub_info_resp)

                    _etapa = "cancel"
                    cancel_resp = await client.post(
                        f"{PAYPAL_API_BASE}/v1/billing/subscriptions/{subscription_id}/cancel",
                        headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"},
                        json={"reason": "El usuario solicitó la cancelación desde la App"}
                    )

                    # [P1-BILLING-CANCEL-FAIL-LOUD · 2026-05-12] Pre-fix esta
                    # rama solo emitía `logger.error` cuando cancel_resp era
                    # != 204/200 y SEGUÍA al UPDATE BD=CANCELLED. PayPal seguía
                    # cobrando mientras el cliente veía "cancelado" en la UI.
                    # Ahora fail-loud: 404 / 422-already-cancelled son
                    # idempotentes (la sub ya está terminal); cualquier otro
                    # status aborta con 502 + alert.
                    if not _is_paypal_cancel_idempotent_success(cancel_resp.status_code, cancel_resp.text):
                        _persist_billing_alert(
                            alert_key=f"billing_cancel_failed:{user_id}:{subscription_id}",
                            severity="critical",
                            title="Cancel: PayPal rechazó la cancelación — riesgo cobro post-cancel",
                            message=(
                                f"User {user_id} solicitó cancel de sub "
                                f"{subscription_id}. PayPal /cancel respondió "
                                f"status={cancel_resp.status_code}: "
                                f"{(cancel_resp.text or '')[:300]}. UPDATE de "
                                f"subscription_status ABORTADO para evitar "
                                f"estado divergente (BD CANCELLED + PayPal "
                                f"sigue cobrando)."
                            ),
                            metadata={
                                "user_id": user_id,
                                "sub_id": subscription_id,
                                "paypal_status": cancel_resp.status_code,
                                "stage": "cancel",
                            },
                        )
                        logger.error(
                            f"❌ [P1-BILLING-CANCEL-FAIL-LOUD] Cancel de sub "
                            f"{subscription_id} falló (status="
                            f"{cancel_resp.status_code}): "
                            f"{(cancel_resp.text or '')[:200]}. Abortando — BD "
                            f"mantiene estado ACTIVE para reflejar realidad."
                        )
                        raise HTTPException(
                            status_code=502,
                            detail=(
                                "El proveedor de pagos rechazó la cancelación. "
                                "Por favor reintenta en unos minutos; si "
                                "persiste, contacta soporte."
                            ),
                        )
                    paypal_consultado = True

                    # [P1-PLAN-LOTE-714] Si el GET de antes no dio la suscripción, un intento más ya cancelada (el
                    # último pago sigue ahí). Best-effort: la cancelación YA ocurrió y no se deshace por esto.
                    if sub_snapshot is None:
                        try:
                            sub_info_resp = await client.get(
                                f"{PAYPAL_API_BASE}/v1/billing/subscriptions/{subscription_id}",
                                headers={"Authorization": f"Bearer {access_token}"}
                            )
                            if sub_info_resp.status_code == 200:
                                sub_snapshot = _json_object_or_none(sub_info_resp)
                        except httpx.HTTPError as _get_e:
                            logger.warning(
                                f"[P1-PLAN-LOTE-714] GET de la sub {subscription_id} tras cancelarla falló: "
                                f"{type(_get_e).__name__}"
                            )
            except (httpx.HTTPError, ValueError) as e:
                _persist_billing_alert(
                    alert_key=f"billing_cancel_failed:{user_id}:{subscription_id}",
                    severity="critical",
                    title="Cancel: PayPal no respondió (timeout/red) — cancelación sin confirmar",
                    message=(
                        f"User {user_id} solicitó cancel de sub {subscription_id}. La llamada a PayPal falló en la "
                        f"etapa '{_etapa}' con {type(e).__name__}: {str(e)[:300]}. UPDATE de subscription_status "
                        f"ABORTADO. Si la etapa es 'cancel', PayPal pudo haberla cancelado igual: verificar en el "
                        f"dashboard (el webhook CANCELLED la reflejará) o esperar el reintento del usuario."
                    ),
                    metadata={
                        "user_id": user_id,
                        "sub_id": subscription_id,
                        "stage": _etapa,
                        "error": type(e).__name__,
                    },
                )
                logger.error(
                    f"❌ [P1-PLAN-LOTE-714] PayPal no respondió al cancelar {subscription_id} "
                    f"(etapa={_etapa}, {type(e).__name__}). Abortando — BD mantiene su estado."
                )
                raise HTTPException(
                    status_code=502,
                    detail=(
                        "No se pudo contactar el proveedor de pagos. "
                        "Por favor reintenta en unos minutos; si "
                        "persiste, contacta soporte."
                    ),
                )
        else:
            logger.warning(
                "⚠️ [DEV-BYPASS] PayPal env vars ausentes; /cancel marca la BD sin "
                "notificar a PayPal (sandbox/MEALFIT_ALLOW_PAYPAL_BYPASS). NO en producción."
            )

        # [P1-PLAN-LOTE-714] Hasta cuándo pagó: `next_billing_time`, o el último pago + un intervalo del plan.
        # La fecha que QUEDA en la BD (puede ser la menor, ver `_mark_subscription_cancelled`) es la que se devuelve.
        hasta, fuente = _paid_through(sub_snapshot)
        filas = await _run_sync_db_in_thread(
            lambda: _mark_subscription_cancelled(where_column="id", where_value=user_id, hasta=hasta)
        )
        if filas:
            access_until = _as_utc_iso(filas[0].get("subscription_end_date"))
        else:
            logger.warning(
                f"[P1-PLAN-LOTE-714] /cancel: la sub {subscription_id} se canceló en PayPal pero el perfil "
                f"{user_id} ya no existe (0 filas)."
            )
            access_until = hasta.isoformat() if hasta is not None else None
        if hasta is None and paypal_consultado and filas:
            await _run_sync_db_in_thread(
                lambda: _alert_access_end_unknown(filas=filas, subscription_id=subscription_id, origen="cancel")
            )
        _invalidate_paypal_status_cache(subscription_id)

        logger.info(
            f"✅ Suscripción {subscription_id} de usuario {user_id} cancelada. Acceso hasta "
            f"{access_until or 'ya (sin fecha de fin)'} (fuente={fuente})."
        )
        return {
            "success": True,
            "message": _CANCEL_OK_MESSAGE if _is_future(access_until) else _CANCEL_OK_SIN_FECHA_MESSAGE,
            "access_until": access_until,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error en /api/subscription/cancel: {str(e)}")
        raise HTTPException(status_code=500, detail=safe_error_detail(e))

@webhooks_router.post("/paypal")
async def api_webhook_paypal(
    request: Request,
    _rl: Optional[str] = Depends(_PAYPAL_WEBHOOK_LIMITER),
):
    try:
        PAYPAL_CLIENT_ID = os.environ.get("PAYPAL_CLIENT_ID")
        PAYPAL_SECRET = os.environ.get("PAYPAL_SECRET")
        PAYPAL_WEBHOOK_ID = os.environ.get("PAYPAL_WEBHOOK_ID")
        is_sandbox = not is_production()  # [P2-PROD-AUDIT-3] SSOT normalizado (lower+strip)
        PAYPAL_API_BASE = "https://api-m.sandbox.paypal.com" if is_sandbox else "https://api-m.paypal.com"
        
        body = await request.body()
        headers = request.headers
        
        # [P2-WEBHOOK-FAIL-SECURE-ALWAYS · 2026-05-12] Pre-fix: si faltaba
        # `PAYPAL_WEBHOOK_ID`/`PAYPAL_CLIENT_ID`/`PAYPAL_SECRET`, el handler
        # SALTABA la verificación de firma y procesaba el evento igual EN
        # SANDBOX (rama `if not is_sandbox: raise`). Si sandbox quedaba
        # accidentalmente expuesto a tráfico real (DNS misroute, deploy con
        # `.env` de prod marcado como sandbox), un atacante podía forge
        # eventos `BILLING.SUBSCRIPTION.SUSPENDED` arbitrarios para downgrade
        # de cualquier usuario via `paypal_subscription_id` enumerado.
        #
        # Ahora fail-secure SIEMPRE: 503 sin importar `is_sandbox`. Escape
        # hatch explícito para dev local sin credenciales reales:
        #   `MEALFIT_ALLOW_WEBHOOK_UNSIGNED=1` (default off).
        # En `ENVIRONMENT=production` el knob NO se respeta — defensa de
        # último recurso si alguien lo flippa por error.
        if not PAYPAL_WEBHOOK_ID or not PAYPAL_CLIENT_ID or not PAYPAL_SECRET:
            allow_unsigned = (
                os.environ.get("MEALFIT_ALLOW_WEBHOOK_UNSIGNED", "").lower()
                in ("1", "true", "yes", "on")
            )
            # En producción el knob se ignora (fail-secure absoluto).
            if not is_sandbox or not allow_unsigned:
                logger.error(
                    "❌ [P2-WEBHOOK-FAIL-SECURE-ALWAYS] PAYPAL_WEBHOOK_ID/"
                    "CLIENT_ID/SECRET ausentes. Rechazando webhook para "
                    "evitar procesar eventos no firmados. Knob "
                    "MEALFIT_ALLOW_WEBHOOK_UNSIGNED solo para dev local; "
                    "ignorado en producción."
                )
                raise HTTPException(
                    status_code=503,
                    detail="Webhook provider misconfigured. Contact support.",
                )
            logger.warning(
                "⚠️ [DEV-BYPASS] PAYPAL_WEBHOOK_ID/keys ausentes; "
                "saltando verificación de firma porque "
                "MEALFIT_ALLOW_WEBHOOK_UNSIGNED=1 + sandbox. NO en producción."
            )
                
        payload_dict = json.loads(body.decode('utf-8'))

        # [P1-PLAN-LOTE-714] El token de la verificación de firma se reutiliza más abajo (GET de la sub para saber
        # hasta cuándo pagó, o para verificar el monto de un huérfano). Sin credenciales (solo dev) no hay token.
        access_token = None
        if PAYPAL_CLIENT_ID and PAYPAL_SECRET and PAYPAL_WEBHOOK_ID:
            async with httpx.AsyncClient(timeout=_HTTPX_TIMEOUT_S) as client:
                auth_resp = await client.post(
                    f"{PAYPAL_API_BASE}/v1/oauth2/token",
                    auth=(PAYPAL_CLIENT_ID, PAYPAL_SECRET),
                    data={"grant_type": "client_credentials"}
                )
                if auth_resp.status_code != 200:
                    # [P2-WEBHOOK-INFRA-503 · 2026-05-30] Fallo TRANSITORIO de infra
                    # (OAuth con PayPal caído/5xx/timeout): NO podemos verificar la
                    # firma, así que NO procesamos — pero devolvemos 5xx para que
                    # PayPal REINTENTE dentro de su ventana de 3 días (25 reintentos).
                    # Pre-fix retornaba HTTP 200 → PayPal lo leía como ack → NO
                    # reintentaba → un SUSPENDED/EXPIRED legítimo se perdía → un
                    # no-pagador quedaba en tier pagado. Distinto de firma inválida
                    # (abajo), que SÍ debe seguir devolviendo 200 (reintentar no ayuda).
                    logger.error("Error autenticando con PayPal en webhook (infra) — devolviendo 503 para retry")
                    raise HTTPException(status_code=503, detail="PayPal auth transient failure; retry.")
                
                access_token = auth_resp.json().get("access_token")
                
                verify_payload = {
                    "auth_algo": headers.get("paypal-auth-algo"),
                    "cert_url": headers.get("paypal-cert-url"),
                    "transmission_id": headers.get("paypal-transmission-id"),
                    "transmission_sig": headers.get("paypal-transmission-sig"),
                    "transmission_time": headers.get("paypal-transmission-time"),
                    "webhook_id": PAYPAL_WEBHOOK_ID,
                    "webhook_event": payload_dict
                }
                
                verify_resp = await client.post(
                    f"{PAYPAL_API_BASE}/v1/notifications/verify-webhook-signature",
                    headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"},
                    json=verify_payload
                )
                
                if verify_resp.status_code != 200 or verify_resp.json().get("verification_status") != "SUCCESS":
                    logger.warning(f"🔒 Bloqueado: Firma del Webhook PayPal inválida: {verify_resp.text}")
                    return {"success": False, "message": "Signature mismatch"}

        # [P2-WEBHOOK-IDEMPOTENCY · 2026-05-30] Dedup de reentrega. PayPal reintenta
        # el MISMO evento hasta 25× en 3 días (y, ahora que devolvemos 503 en fallos
        # de infra, esos reintentos son esperados). Sin dedup, cada reentrega
        # re-ejecuta el UPDATE — la mayoría idempotentes en valor, pero amplifica
        # cualquier transición de estado. INSERT ON CONFLICT DO NOTHING sobre
        # `transmission_id` (PK): si 0 filas → ya procesado → ack sin re-mutar.
        # Best-effort: si la tabla/insert falla, seguimos (degrada al comportamiento
        # previo, no bloquea webhooks legítimos). Solo corre tras pasar la firma.
        # [P1-WEBHOOK-DEDUP-ATOMIC · 2026-06-19] (audit fresco P1-2) Claim de DOS FASES para que el dedup no
        # PIERDA eventos de billing. Antes el marcador se INSERTaba+commiteaba (su propia conexión autocommit)
        # ANTES de aplicar la transición de estado (UPDATE de plan_tier/subscription_status): si un UPDATE fallaba
        # transitoriamente (hiccup de Neon/lock/red) → `except` → 503 → PayPal reintenta el MISMO transmission_id,
        # pero el reintento veía el marcador YA committeado y la rama `deduped` SALTABA el procesamiento → el
        # evento se perdía PERMANENTEMENTE (no-pagador en tier pagado / pagador degradado). Reabría exactamente la
        # clase que P2-WEBHOOK-INFRA-503 cerró, vía la capa de dedup. Ahora el marcador solo cuenta como
        # "procesado" cuando su value es {"status":"done"}, escrito DESPUÉS del procesamiento exitoso (fase 2, al
        # final del try). Un marcador 'processing' (claim cuya entrega previa murió a mitad) NO deduplica → el
        # reintento re-procesa (los UPDATE de estado son idempotentes en valor). El prefijo `paypal_webhook:`
        # lo sigue limpiando el sweep `_KV_SWEEP_PREFIXES`.
        _transmission_id = headers.get("paypal-transmission-id")
        _dedup_kv_key = f"paypal_webhook:{_transmission_id}" if _transmission_id else None
        if _dedup_kv_key:
            try:
                # Fase 1 (claim): INSERT 'processing' ON CONFLICT DO NOTHING RETURNING. Lista no-vacía = lo
                # claimeamos nosotros (primera entrega). Lista vacía = el marcador ya existía → deduplicar SOLO
                # si una entrega previa COMPLETÓ (status=done); si está 'processing', re-procesar (no perder).
                _claimed = await asyncio.to_thread(
                    execute_sql_write,
                    "INSERT INTO app_kv_store (key, value) VALUES (%s, jsonb_build_object('status', 'processing')) "
                    "ON CONFLICT (key) DO NOTHING RETURNING key",
                    (_dedup_kv_key,), returning=True,
                )
                if not _claimed:
                    _already_done = await asyncio.to_thread(
                        execute_sql_write,
                        "SELECT 1 AS done FROM app_kv_store WHERE key = %s AND value->>'status' = 'done'",
                        (_dedup_kv_key,), returning=True,
                    )
                    if _already_done:
                        logger.info(
                            f"♻️ [P2-WEBHOOK-IDEMPOTENCY] transmission_id={_transmission_id} "
                            "ya procesado (status=done); ack idempotente sin re-procesar."
                        )
                        return {"success": True, "deduped": True}
                    logger.info(
                        f"♻️ [P1-WEBHOOK-DEDUP-ATOMIC] transmission_id={_transmission_id} con marcador "
                        "'processing' (entrega previa incompleta); re-procesando para no perder el evento."
                    )
            except Exception as _dedup_e:
                logger.warning(
                    f"[P2-WEBHOOK-IDEMPOTENCY] dedup best-effort falló ({_dedup_e}); "
                    "procesando el evento de todos modos."
                )
                _dedup_kv_key = None  # KV roto → no intentar marcar 'done' al final

        event_type = payload_dict.get("event_type")
        resource = payload_dict.get("resource", {})

        logger.info(f"⚡ [WEBHOOK PAYPAL] Evento recibido: {event_type}")

        # [P1-PLAN-LOTE-714] Cualquier evento de una suscripción deja viejo lo que `/status` tuviera cacheado de ella.
        if isinstance(resource, dict):
            _invalidate_paypal_status_cache(resource.get("id"), resource.get("billing_agreement_id"))
        
        # [P2-BILLING-PAYMENT-FAILED-GRACE · 2026-05-30] `PAYMENT.FAILED` ya NO es
        # terminal. Pre-fix lo incluía en `downgrade_events` → un fallo de pago
        # TRANSITORIO (la tarjeta rebota una vez; PayPal reintenta durante su
        # dunning window de varios días) degradaba al usuario a `gratis`
        # INSTANTÁNEAMENTE, y NO existía handler de re-activación → el usuario
        # perdía el acceso pagado AUNQUE PayPal cobrara con éxito en el reintento.
        # Ahora:
        #   - PAYMENT.FAILED → status no-destructivo `PAYMENT_RETRYING` (conserva
        #     `plan_tier`; el acceso se gobierna por plan_tier, no por status).
        #   - Solo SUSPENDED/EXPIRED (+ CANCELLED abajo) degradan (terminales).
        #   - ACTIVATED / PAYMENT.SALE.COMPLETED → re-activa: restaura el tier
        #     desde el `plan_id` de PayPal y limpia el flag de retry.
        # Knob `MEALFIT_BILLING_PAYMENT_FAILED_GRACE` (default True) = kill-switch:
        # a False revierte al comportamiento legacy (PAYMENT.FAILED degrada).
        # Tooltip-anchor: P2-BILLING-PAYMENT-FAILED-GRACE.
        _payment_failed_grace = _env_bool("MEALFIT_BILLING_PAYMENT_FAILED_GRACE", True)

        downgrade_events = [
            "BILLING.SUBSCRIPTION.SUSPENDED",
            "BILLING.SUBSCRIPTION.EXPIRED",
        ]
        if not _payment_failed_grace:
            # Rollback legacy: tratar PAYMENT.FAILED como terminal.
            downgrade_events.append("BILLING.SUBSCRIPTION.PAYMENT.FAILED")

        if event_type in downgrade_events:
            subscription_id = resource.get("id")
            if subscription_id:
                logger.info(f"⬇️ Degradando suscripción {subscription_id} en BD debido a {event_type}.")
                await _run_sync_db_in_thread(
                    lambda: execute_sql_write(
                        "UPDATE public.user_profiles SET plan_tier = 'gratis', "
                        "subscription_status = 'INACTIVE' WHERE paypal_subscription_id = %s",
                        (subscription_id,),
                    )
                )
        elif _payment_failed_grace and event_type == "BILLING.SUBSCRIPTION.PAYMENT.FAILED":
            # Fallo de pago transitorio: NO degradar. Marcar retry y conservar tier.
            subscription_id = resource.get("id")
            if subscription_id:
                logger.warning(
                    f"💳 [P2-BILLING-PAYMENT-FAILED-GRACE] PAYMENT.FAILED para sub "
                    f"{subscription_id}: marcando PAYMENT_RETRYING SIN degradar tier "
                    f"(ventana de reintento de PayPal). Solo SUSPENDED/EXPIRED degradan."
                )
                await _run_sync_db_in_thread(
                    lambda: execute_sql_write(
                        "UPDATE public.user_profiles SET subscription_status = 'PAYMENT_RETRYING' "
                        "WHERE paypal_subscription_id = %s",
                        (subscription_id,),
                    )
                )
        elif event_type in ("BILLING.SUBSCRIPTION.ACTIVATED", "PAYMENT.SALE.COMPLETED"):
            # Re-activación / pago exitoso (posiblemente tras un PAYMENT.FAILED).
            # ACTIVATED es un recurso de SUSCRIPCIÓN (id + plan_id → restaura tier);
            # PAYMENT.SALE.COMPLETED es un recurso de VENTA (la suscripción está en
            # `billing_agreement_id`, sin plan_id → solo limpia el flag de retry).
            if event_type == "BILLING.SUBSCRIPTION.ACTIVATED":
                subscription_id = resource.get("id")
                plan_id = resource.get("plan_id")
            else:  # PAYMENT.SALE.COMPLETED
                subscription_id = resource.get("billing_agreement_id")
                plan_id = None
            restored_tier = _build_paypal_plan_tier_map().get(plan_id) if plan_id else None
            if subscription_id:
                # [P1-BILLING-REACTIVATE-NOT-CANCELLED · 2026-05-30] NUNCA reactivar
                # una suscripción CANCELLED. PayPal NO garantiza orden de webhooks
                # (reintenta hasta 3 días): un PAYMENT.SALE.COMPLETED tardío del
                # ciclo pagado puede aterrizar DESPUÉS de un CANCELLED. Pre-fix esto
                # flippeaba `subscription_status` a ACTIVE INCONDICIONALMENTE →
                # acceso pagado perpetuo sin cobro, porque el único revocador
                # (db_profiles.py: degradación post-fin-de-ciclo) exige
                # status==CANCELLED y jamás dispara tras la resurrección. Regresión
                # introducida por el propio P2-BILLING-PAYMENT-FAILED-GRACE (mismo
                # día). Knob kill-switch revierte al flip incondicional legacy.
                _reactivate_guard = _env_bool("MEALFIT_BILLING_REACTIVATE_NOT_CANCELLED", True)
                update_payload = {"subscription_status": "ACTIVE"}
                if restored_tier:
                    update_payload["plan_tier"] = restored_tier
                logger.info(
                    f"⬆️ [P2-BILLING-PAYMENT-FAILED-GRACE] {event_type} sub {subscription_id}: "
                    f"status→ACTIVE"
                    + (f", tier→{restored_tier}" if restored_tier else " (tier sin cambio)")
                )

                def _do_reactivate():
                    # [P1-NEON-DB-MIGRATION · 2026-06-12] SET dinámico desde
                    # `update_payload` (status siempre; plan_tier solo si se
                    # derivó de plan_id) + filtro condicional del guard.
                    set_clauses = ["subscription_status = %s"]
                    params: list = [update_payload["subscription_status"]]
                    if "plan_tier" in update_payload:
                        set_clauses.append("plan_tier = %s")
                        params.append(update_payload["plan_tier"])
                    # [P1-PLAN-LOTE-714 · 2026-09-28] Una sub que vuelve a estar ACTIVE no tiene fecha de fin: la que
                    # quedara de una cancelación anterior se pintaba como «Próximo cobro» en Configuración. Con el
                    # guard de abajo esto nunca toca una CANCELLED (esa conserva su fecha de gracia).
                    # tooltip-anchor: P1-PLAN-LOTE-714-END-DATE-CLEAR
                    set_clauses.append("subscription_end_date = NULL")
                    where_clauses = ["paypal_subscription_id = %s"]
                    params.append(subscription_id)
                    if _reactivate_guard:
                        if event_type == "PAYMENT.SALE.COMPLETED":
                            # Pago dentro del ciclo (sin plan_id): SOLO limpiar el
                            # flag PAYMENT_RETRYING → ACTIVE. No forzar ACTIVE sobre
                            # CANCELLED/INACTIVE (esos los gobierna su propia lógica).
                            where_clauses.append("subscription_status = 'PAYMENT_RETRYING'")
                        else:
                            # ACTIVATED puede reactivar PAYMENT_RETRYING/INACTIVE pero
                            # NUNCA una sub CANCELLED por el usuario. `<>` no matchea
                            # filas con status NULL — mismo comportamiento que el
                            # `.neq()` de PostgREST que reemplaza.
                            where_clauses.append("subscription_status <> 'CANCELLED'")
                    # [P1-BILLING-ORPHAN-RECOVERY · 2026-08-22] `RETURNING id` para
                    # distinguir "aplicado" de "no matchó ninguna fila". Sin esta
                    # señal el no-op era indistinguible del éxito, que es
                    # exactamente cómo se perdían los cobros huérfanos.
                    return execute_sql_write(
                        f"UPDATE public.user_profiles SET {', '.join(set_clauses)} "
                        f"WHERE {' AND '.join(where_clauses)} RETURNING id",
                        tuple(params),
                        returning=True,
                    )

                _reactivated_rows = await _run_sync_db_in_thread(_do_reactivate)

                # [P1-BILLING-ORPHAN-RECOVERY · 2026-08-22] 0 filas en un ACTIVATED
                # puede significar que NADIE tiene esta sub: el `/verify` del
                # navegador no llegó y PayPal ya está cobrando. Se intenta atribuir
                # por el `custom_id` que el propio checkout estampó.
                #
                # SOLO `ACTIVATED`: `PAYMENT.SALE.COMPLETED` no trae `plan_id` (no hay
                # tier que derivar) y su guard es `PAYMENT_RETRYING`-only, así que 0
                # filas es su caso NORMAL — llamar aquí convertiría cada pago de
                # renovación en una alerta falsa.
                if (
                    not _reactivated_rows
                    and event_type == "BILLING.SUBSCRIPTION.ACTIVATED"
                    and _env_bool("MEALFIT_BILLING_ORPHAN_RECOVERY", True)
                ):
                    # [P1-PLAN-LOTE-714] Async: antes de adoptar corre la verificación de monto de `/verify`, que
                    # habla con PayPal (lista de precios) — ver `_recover_orphan_subscription`.
                    await _recover_orphan_subscription(
                        subscription_id=subscription_id,
                        custom_id=resource.get("custom_id"),
                        tier=restored_tier,
                        plan_id=plan_id,
                        resource=resource,
                        access_token=access_token,
                        paypal_api_base=PAYPAL_API_BASE,
                    )
        elif event_type == "BILLING.SUBSCRIPTION.CANCELLED":
            subscription_id = resource.get("id")

            if subscription_id:
                logger.info(f"ℹ️ Webhook: Suscripción {subscription_id} cancelada remota/silenciosamente.")

                # [P1-PLAN-LOTE-714 · 2026-09-28] El recurso de una sub CANCELLED casi nunca trae `next_billing_time`
                # (ya no hay próximo cobro), y escribir solo el estado degradaba al usuario AL INSTANTE aunque hubiera
                # pagado el mes. La fecha sale del propio recurso (firmado): `next_billing_time` o el último pago + un
                # intervalo del plan; si al recurso le falta, de un GET a PayPal con el token de la firma. Solo si no
                # hay forma de saberlo se escribe el estado a secas, como antes, y se alerta.
                _hasta, _fuente = _paid_through(resource)
                if _hasta is None and access_token:
                    _hasta, _fuente = _paid_through(
                        await _fetch_paypal_subscription(PAYPAL_API_BASE, access_token, subscription_id)
                    )
                _wh_sub_id, _wh_hasta = subscription_id, _hasta
                _filas_canceladas = await _run_sync_db_in_thread(
                    lambda: _mark_subscription_cancelled(
                        where_column="paypal_subscription_id", where_value=_wh_sub_id, hasta=_wh_hasta,
                    )
                )
                if _hasta is None and _filas_canceladas:
                    await _run_sync_db_in_thread(
                        lambda: _alert_access_end_unknown(
                            filas=_filas_canceladas, subscription_id=_wh_sub_id, origen="webhook CANCELLED",
                        )
                    )
                logger.info(
                    f"ℹ️ [P1-PLAN-LOTE-714] sub {subscription_id} CANCELLED: fin de acceso "
                    f"{_hasta.isoformat() if _hasta else 'indeterminado'} (fuente={_fuente}); "
                    f"{len(_filas_canceladas)} perfil(es)."
                )

        # [P1-WEBHOOK-DEDUP-ATOMIC · 2026-06-19] (audit fresco P1-2) Fase 2: marca el evento COMPLETADO solo
        # ahora que TODAS las transiciones de estado se aplicaron sin excepción. Si un UPDATE de arriba hubiera
        # lanzado, el flujo saltó al `except` (→ 503) SIN llegar aquí → el marcador queda 'processing' → el
        # reintento de PayPal re-procesa (en vez de deduplicar y perder el evento). Best-effort: si el marcado
        # falla, el reintento re-procesa (idempotente), nunca pierde el evento.
        if _dedup_kv_key:
            try:
                await asyncio.to_thread(
                    execute_sql_write,
                    # `updated_at = NOW()` igual que los demás writers de app_kv_store: el GC (`_KV_SWEEP_PREFIXES`)
                    # borra por updated_at, así el TTL del marcador se cuenta desde el completion, no desde el claim.
                    "UPDATE app_kv_store SET value = jsonb_build_object('status', 'done'), updated_at = NOW() "
                    "WHERE key = %s",
                    (_dedup_kv_key,),
                )
            except Exception as _mark_e:
                logger.warning(
                    f"[P1-WEBHOOK-DEDUP-ATOMIC] no se pudo marcar 'done' "
                    f"({type(_mark_e).__name__}); el reintento re-procesará (idempotente)."
                )

        return {"success": True}

    except HTTPException:
        # [P2-WEBHOOK-INFRA-503 · 2026-05-30] Propagar las HTTPException intencionales
        # (503 fail-secure de misconfig @705 + 503 infra-OAuth). Pre-fix el
        # `except Exception` de abajo las TRAGABA (HTTPException ⊂ Exception) y las
        # convertía en HTTP 200 → el fail-secure 503 nunca llegaba al cliente y
        # PayPal no reintentaba. Sin este `raise`, ambos 503 quedaban neutralizados.
        raise
    except Exception as e:
        # [P2-WEBHOOK-INFRA-503 · 2026-05-30] Cualquier otro fallo transitorio de
        # procesamiento (red, timeout de verify, parse): devolver 5xx para que
        # PayPal reintente, en vez de descartar el evento con un 200 silencioso.
        # Firma genuinamente inválida sigue retornando 200 (es un `return`, no llega
        # aquí). Un bug determinista reintentará 25× y luego PayPal desiste — ruido
        # acotado, preferible a perder un evento de billing legítimo.
        logger.error(f"❌ Error procesando webhook PayPal: {e}")
        raise HTTPException(status_code=503, detail="Webhook processing transient error; retry.")
