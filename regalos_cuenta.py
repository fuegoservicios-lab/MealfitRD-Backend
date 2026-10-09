"""[P1-PLAN-LOTE-771 · 2026-09-28] Regalos de la cuenta: créditos extra y planes de cortesía.

Spec: docs/superpowers/specs/2026-09-28-admin-cuentas-regalos-design.md (raíz del workspace), §3.

Lo pagado y lo regalado NUNCA se mezclan. PayPal escribe `user_profiles.plan_tier`; este módulo lee
`public.account_grants` y SUPERPONE al leer: el plan que cuenta es el mayor de los dos y la cortesía caduca sola (no
hay cron). Un regalo de créditos no borra uso —el panel de costes sigue siendo verdad—: sube el tope.

Fail-open hacia lo pagado: si la tabla no se puede leer o el knob `MEALFIT_ACCOUNT_GRANTS` está apagado, cada cuenta
queda exactamente con lo que paga.
"""
from __future__ import annotations

import logging
import re
from datetime import date, datetime, timedelta, timezone

from db import execute_sql_query
from knobs import _env_bool

logger = logging.getLogger(__name__)

RANGO = {"gratis": 0, "basic": 1, "plus": 2, "ultra": 3}
PLANES_REGALABLES = ("basic", "plus", "ultra")
MEDIDORES = {"generacion": "creditos_generacion", "coach": "creditos_coach"}
DIAS_AVISO = 14
_ES_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)
_VIGENTE = "revoked_at IS NULL AND starts_at <= now() AND (ends_at IS NULL OR ends_at > now())"


def activo() -> bool:
    """Kill switch: en False la superposición y los topes extra se ignoran y el panel no regala."""
    return _env_bool("MEALFIT_ACCOUNT_GRANTS", True)


def iso(v):
    if isinstance(v, (datetime, date)):
        return v.isoformat()
    return str(v) if v else None


def inicio_de_mes(desplazamiento: int = 0, ahora: datetime | None = None) -> datetime:
    """Día 1 a las 00:00 UTC del mes actual + `desplazamiento`. SSOT de la ventana mensual: la usan el contador de
    créditos (`get_monthly_api_usage`) y la caducidad de los regalos, así un regalo «de este mes» se acaba justo
    cuando el contador se reinicia."""
    ahora = ahora or datetime.now(timezone.utc)
    indice = ahora.year * 12 + (ahora.month - 1) + desplazamiento
    return datetime(indice // 12, indice % 12 + 1, 1, tzinfo=timezone.utc)


def regalos_vigentes(user_id, usage_scope="web") -> list:
    """Los regalos vigentes de la cuenta. [] si el knob está apagado, el id no es de una cuenta o la lectura falla."""
    uid = str(user_id or "")
    if not _ES_UUID.match(uid) or not activo():
        return []
    try:
        return execute_sql_query(
            "SELECT id::text AS id, kind, amount, plan, starts_at, ends_at, created_at FROM public.account_grants "
            f"WHERE user_id = %s AND usage_scope = %s AND {_VIGENTE} ORDER BY created_at",
            (uid, usage_scope), fetch_all=True) or []
    except Exception as e:  # noqa: BLE001 — sin regalos queda lo pagado, nunca menos
        logger.warning(f"⚠️ [P1-PLAN-LOTE-771] regalos no legibles para {uid}: {e!r}")
        return []


def extra_de(regalos, medidor: str) -> int:
    tipo = MEDIDORES[medidor]
    return sum(int(r.get("amount") or 0) for r in regalos or [] if r.get("kind") == tipo)


def cortesia_de(regalos):
    """La cortesía de plan vigente (la más reciente), o None."""
    for r in reversed(list(regalos or [])):
        if r.get("kind") == "plan" and r.get("plan") in PLANES_REGALABLES:
            return {"plan": r["plan"], "hasta": r.get("ends_at")}
    return None


def plan_efectivo(pagado, cortesia):
    """El plan que la persona disfruta: el mayor entre lo pagado y la cortesía. `admin` no se toca."""
    if pagado == "admin" or not cortesia:
        return pagado
    if RANGO.get(cortesia.get("plan"), -1) > RANGO.get(pagado or "gratis", 0):
        return cortesia["plan"]
    return pagado


def superponer(perfil):
    """El perfil con el plan que la persona DISFRUTA. Añade `plan_tier_pagado` (lo de PayPal: las pantallas de cobro
    deciden con él), `cortesia` ({plan, hasta} solo si está en efecto) y `creditos_extra` ({generacion, coach}).
    Jamás se escribe de vuelta."""
    if not perfil:
        return perfil
    from ios_free import is_free, project_profile
    if is_free():
        return project_profile(perfil)
    pagado = perfil.get("plan_tier")
    regalos = [] if pagado == "admin" else regalos_vigentes(perfil.get("id"))
    cortesia = cortesia_de(regalos)
    efectivo = plan_efectivo(pagado, cortesia)
    perfil["plan_tier_pagado"] = pagado
    perfil["plan_tier"] = efectivo
    perfil["cortesia"] = (
        {"plan": cortesia["plan"], "hasta": iso(cortesia["hasta"])}
        if cortesia and efectivo == cortesia["plan"] and efectivo != pagado else None)
    perfil["creditos_extra"] = {"generacion": extra_de(regalos, "generacion"), "coach": extra_de(regalos, "coach")}
    return perfil


def resumen_creditos(perfil) -> dict:
    """Para `GET /api/user/credits`: el tope REAL de créditos de planes (plan efectivo + regalos), cuánto es regalo y
    hasta cuándo, y los regalos de los últimos `DIAS_AVISO` días que están en efecto, para avisar a la persona.
    Recibe el perfil YA superpuesto (`get_user_profile`)."""
    from auth import _TIER_LIMITS   # perezoso: auth importa db al arrancar y este módulo lo importa db_profiles
    perfil = perfil or {}
    tier = perfil.get("plan_tier") or "gratis"
    from ios_free import is_free, GENERATION
    base = GENERATION if is_free() else int(_TIER_LIMITS.get(tier, _TIER_LIMITS["gratis"]))
    regalos = (regalos_vigentes(perfil.get("id"), usage_scope="ios_free") if is_free()
               else [] if tier == "admin" else regalos_vigentes(perfil.get("id")))
    extra = extra_de(regalos, "generacion")
    hastas = [r["ends_at"] for r in regalos if r.get("kind") == MEDIDORES["generacion"] and r.get("ends_at")]
    desde = datetime.now(timezone.utc) - timedelta(days=DIAS_AVISO)
    # Revisión final (Review Focus 3): una cortesía se anuncia solo si está EN EFECTO (`superponer` la deja en
    # `cortesia`); igualar el plan efectivo no basta: cortesía Plus y luego paga Plus diría «Plus de cortesía».
    en_efecto = (perfil.get("cortesia") or {}).get("plan")
    recientes = [
        {"id": r["id"], "tipo": r["kind"], "cantidad": r.get("amount"), "plan": r.get("plan"),
         "hasta": iso(r.get("ends_at"))}
        for r in regalos
        if isinstance(r.get("created_at"), datetime) and r["created_at"] >= desde
        and (r.get("kind") != "plan" or (en_efecto is not None and r.get("plan") == en_efecto))]
    return {"limit": base + extra, "bonus": extra, "bonus_hasta": iso(max(hastas)) if hastas else None,
            "regalos_recientes": recientes}
