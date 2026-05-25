import asyncio
import os
import json
import logging
import httpx
from fastapi import APIRouter, Body, Depends, HTTPException, Request
from error_utils import safe_error_detail
from typing import Optional

# Imports relativos al backend
from auth import get_verified_user_id
from db import supabase
from knobs import _env_float
from rate_limiter import RateLimiter


# [P1-ASYNC-SYNC-DB-BLOCKING · 2026-05-24] Helper SSOT para despachar calls
# sync del cliente Supabase desde handlers `async def`. Pre-fix los 6
# callsites `supabase.table(...).execute()` corrían inline dentro del event
# loop, bloqueando ~10-200ms por roundtrip y throttlando otros handlers
# async bajo carga (chat stream, webhook PayPal, diary upload). Mismo
# patrón que P2-AUTH-ASYNC-SLEEP cerró para `auth.py::get_verified_user_id`.
# `asyncio.to_thread` despacha al default thread pool — el event loop sirve
# otras requests mientras Supabase responde. Tooltip-anchor:
# P1-ASYNC-SYNC-DB-BLOCKING.
async def _supabase_async(thunk):
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

    Idempotente vía `on_conflict='alert_key'`: alerts repetidos del mismo
    `(user_id, sub_id)` actualizan `triggered_at` sin duplicar filas.
    Cualquier excepción se loguea sin propagar — la alert es observabilidad
    y NO debe romper el flujo del handler que ya está mid-failure.

    Modelo de resolution: **Manual** (ver "Política de `system_alerts`
    resolution" en CLAUDE.md). SRE debe verificar PayPal dashboard y
    reconciliar BD manualmente; no hay auto-cierre.
    """
    if supabase is None:
        return
    try:
        from datetime import datetime, timezone
        now_iso = datetime.now(timezone.utc).isoformat()
        supabase.table("system_alerts").upsert({
            "alert_key": alert_key,
            "alert_type": "billing",
            "severity": severity,
            "title": title,
            "message": message,
            "metadata": metadata or {},
            "triggered_at": now_iso,
            "resolved_at": None,
        }, on_conflict="alert_key").execute()
    except Exception as e:
        logger.error(
            f"[P1-BILLING-FAIL-LOUD] No se pudo persistir alert "
            f"{alert_key}: {type(e).__name__}: {e}"
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
    """Valida un código de descuento contra la tabla discount_codes en Supabase.

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

        # Buscar el código en la tabla
        res = await _supabase_async(
            lambda: supabase.table("discount_codes").select("*").eq("code", code).eq("is_active", True).execute()
        )

        if not res.data or len(res.data) == 0:
            return {"valid": False, "message": "Código no encontrado o inactivo."}

        discount = res.data[0]

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
    mapping = {}
    for tier in ("basic", "plus", "ultra"):
        plan_id = os.environ.get(f"PAYPAL_PLAN_{tier.upper()}_ID")
        if plan_id:
            mapping[plan_id] = tier
    return mapping


@router.post("/verify")
async def api_verify_subscription(data: dict = Body(...), verified_user_id: str = Depends(get_verified_user_id)):
    try:
        user_id = data.get("user_id")
        subscription_id = data.get("subscriptionID")
        client_hint_tier = (data.get("tier") or "").strip().lower()

        if not user_id or not subscription_id or not client_hint_tier:
            raise HTTPException(status_code=400, detail="Missing parameters")

        if not verified_user_id or verified_user_id != user_id:
            logger.warning(f"Intento no autorizado de verificar suscripcion: req_user={user_id}, auth_user={verified_user_id}")
            raise HTTPException(status_code=401, detail="No autorizado.")

        PAYPAL_CLIENT_ID = os.environ.get("PAYPAL_CLIENT_ID")
        PAYPAL_SECRET = os.environ.get("PAYPAL_SECRET")
        is_sandbox = os.environ.get("ENVIRONMENT") != "production"
        PAYPAL_API_BASE = "https://api-m.sandbox.paypal.com" if is_sandbox else "https://api-m.paypal.com"

        # [P0-BILLING-2 · 2026-05-12] Fail-secure si faltan env vars PayPal
        # en producción. Pre-fix: cuando ambas keys eran falsy, el handler
        # forzaba `success` a True (bypass completo de verificación). Si el
        # contenedor perdía las env vars (rotación rota, misconfig Easypanel),
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

        if env_ready:
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
                    f"mapea a tier interno. Configurar env vars "
                    f"PAYPAL_PLAN_BASIC_ID/PAYPAL_PLAN_PLUS_ID/"
                    f"PAYPAL_PLAN_ULTRA_ID. cliente hint: {client_hint_tier!r}."
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

        existing_res = await _supabase_async(
            lambda: supabase.table("user_profiles").select("paypal_subscription_id, subscription_status").eq("id", user_id).execute()
        )
        if existing_res.data and len(existing_res.data) > 0:
            old_sub_id = existing_res.data[0].get("paypal_subscription_id")
            old_status = existing_res.data[0].get("subscription_status")

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

        res = await _supabase_async(
            lambda: supabase.table("user_profiles").update({
                "plan_tier": tier_to_assign,
                "paypal_subscription_id": subscription_id,
                "subscription_status": "ACTIVE"
            }).eq("id", user_id).execute()
        )

        return {"success": True, "message": "Suscripción verificada y plan actualizado B2B."}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error interno en /api/subscription/verify: {str(e)}")
        raise HTTPException(status_code=500, detail=safe_error_detail(e))

@router.post("/cancel")
async def api_cancel_subscription(data: dict = Body(...), verified_user_id: str = Depends(get_verified_user_id)):
    try:
        user_id = data.get("user_id")
        
        if not verified_user_id or verified_user_id != user_id:
            logger.warning(f"Intento no autorizado de cancelar suscripcion: req_user={user_id}, auth_user={verified_user_id}")
            raise HTTPException(status_code=401, detail="No autorizado.")
            
        res = await _supabase_async(
            lambda: supabase.table("user_profiles").select("paypal_subscription_id").eq("id", user_id).execute()
        )
        if not res.data or not res.data[0].get("paypal_subscription_id"):
            raise HTTPException(status_code=400, detail="No active subscription found to cancel.")
            
        subscription_id = res.data[0]["paypal_subscription_id"]
        
        PAYPAL_CLIENT_ID = os.environ.get("PAYPAL_CLIENT_ID")
        PAYPAL_SECRET = os.environ.get("PAYPAL_SECRET")
        is_sandbox = os.environ.get("ENVIRONMENT") != "production"
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

        end_date = None

        if env_ready:
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

                sub_info_resp = await client.get(
                    f"{PAYPAL_API_BASE}/v1/billing/subscriptions/{subscription_id}",
                    headers={"Authorization": f"Bearer {access_token}"}
                )
                if sub_info_resp.status_code == 200:
                    billing_info = sub_info_resp.json().get("billing_info", {})
                    end_date = billing_info.get("next_billing_time")

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
        
        logger.info(f"✅ Suscripción {subscription_id} de usuario {user_id} cancelada. Mantendrá acceso hasta {end_date or 'fin de ciclo'}.")
        
        update_payload = {"subscription_status": "CANCELLED"}
        if end_date:
            update_payload["subscription_end_date"] = end_date

        await _supabase_async(
            lambda: supabase.table("user_profiles").update(update_payload).eq("id", user_id).execute()
        )

        return {"success": True, "message": "Tu suscripción no se renovará, pero mantendrás tu plan actual hasta el final del ciclo pagado."}

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
        is_sandbox = os.environ.get("ENVIRONMENT") != "production"
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
        
        if PAYPAL_CLIENT_ID and PAYPAL_SECRET and PAYPAL_WEBHOOK_ID:
            async with httpx.AsyncClient(timeout=_HTTPX_TIMEOUT_S) as client:
                auth_resp = await client.post(
                    f"{PAYPAL_API_BASE}/v1/oauth2/token",
                    auth=(PAYPAL_CLIENT_ID, PAYPAL_SECRET),
                    data={"grant_type": "client_credentials"}
                )
                if auth_resp.status_code != 200:
                    logger.error("Error autenticando con PayPal en webhook")
                    return {"success": False}
                
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
        
        event_type = payload_dict.get("event_type")
        resource = payload_dict.get("resource", {})
        
        logger.info(f"⚡ [WEBHOOK PAYPAL] Evento recibido: {event_type}")
        
        downgrade_events = [
            "BILLING.SUBSCRIPTION.SUSPENDED", 
            "BILLING.SUBSCRIPTION.EXPIRED",
            "BILLING.SUBSCRIPTION.PAYMENT.FAILED"
        ]
        
        if event_type in downgrade_events:
            subscription_id = resource.get("id")
            if subscription_id:
                logger.info(f"⬇️ Degradando suscripción {subscription_id} en BD debido a {event_type}.")
                await _supabase_async(
                    lambda: supabase.table("user_profiles").update({
                        "plan_tier": "gratis",
                        "subscription_status": "INACTIVE"
                    }).eq("paypal_subscription_id", subscription_id).execute()
                )
        elif event_type == "BILLING.SUBSCRIPTION.CANCELLED":
            subscription_id = resource.get("id")
            end_date = resource.get("billing_info", {}).get("next_billing_time")
            
            if subscription_id:
                logger.info(f"ℹ️ Webhook: Suscripción {subscription_id} cancelada remota/silenciosamente.")
                
                update_payload = {"subscription_status": "CANCELLED"}
                if end_date:
                    update_payload["subscription_end_date"] = end_date

                await _supabase_async(
                    lambda: supabase.table("user_profiles").update(update_payload).eq("paypal_subscription_id", subscription_id).execute()
                )
        
        return {"success": True}

    except Exception as e:
        logger.error(f"❌ Error procesando webhook PayPal: {e}")
        return {"success": False}
