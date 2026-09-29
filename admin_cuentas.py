# backend/admin_cuentas.py
"""[P1-PLAN-LOTE-774 · 2026-09-28] Panel admin · Cuentas (spec 2026-09-28-admin-cuentas-regalos-design §4.1).

Buscar una cuenta por su correo EXACTO (no hay lista de cuentas), ver su ficha y regalarle créditos o un plan de
cortesía. Cada cambio se anota en `admin_access_log` ANTES de escribir: si el rastro no se puede anotar, no hay cambio
(503). La ficha no trae datos de salud, conversaciones ni fotos. Lo pagado (`user_profiles.plan_tier`) no se toca
nunca: los regalos viven en `account_grants` y `regalos_cuenta` los superpone al leer.
"""
from __future__ import annotations

import logging
import uuid
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import regalos_cuenta as rc
from admin_acceso import registrar_acceso
from avisos_regalo import avisar_en_segundo_plano
from db import execute_sql_query, execute_sql_transaction, execute_sql_write, get_monthly_api_usage

logger = logging.getLogger(__name__)

_ZONA_RD = ZoneInfo("America/Santo_Domingo")
MAX_CREDITOS = 1000
MAX_DIAS_CORTESIA = 366
_HISTORIAL = 50
_NOMBRE_PLAN = {"gratis": "Gratis", "basic": "Básico", "plus": "Plus", "ultra": "Max", "admin": "Administración"}


class ErrorRegalo(Exception):
    """Un regalo que no se puede dar: el router lo devuelve con su código y su frase."""

    def __init__(self, status: int, detalle: str):
        super().__init__(detalle)
        self.status = status
        self.detalle = detalle


def normalizar_correo(v) -> str:
    return "".join(str(v or "").split()).lower()


def buscar_por_correo(correo):
    c = normalizar_correo(correo)
    if not c or "@" not in c or len(c) > 254:
        return None
    fila = execute_sql_query(
        "SELECT id::text AS id FROM public.user_profiles WHERE lower(email) = %s ORDER BY created_at DESC LIMIT 1",
        (c,), fetch_one=True)
    return (fila or {}).get("id")


def _perfil(user_id):
    return execute_sql_query(
        "SELECT id::text AS id, email, full_name, created_at, plan_tier, subscription_status, subscription_end_date, "
        "paypal_subscription_id IS NOT NULL AS tiene_paypal FROM public.user_profiles WHERE id = %s",
        (user_id,), fetch_one=True)


def _topes(tier) -> tuple:
    from auth import _COACH_LIMITS, _TIER_LIMITS
    return (int(_TIER_LIMITS.get(tier, _TIER_LIMITS["gratis"])),
            int(_COACH_LIMITS.get(tier, _COACH_LIMITS["gratis"])))


def _estado(r, ahora) -> str:
    if r.get("revoked_at"):
        return "revertido"
    if r.get("ends_at") and r["ends_at"] <= ahora:
        return "caducado"
    return "vigente"


def _detalle(r) -> str:
    if r.get("kind") == "plan":
        return f"{_NOMBRE_PLAN.get(r.get('plan'), r.get('plan'))} de cortesía"
    que = "créditos de planes" if r.get("kind") == rc.MEDIDORES["generacion"] else "mensajes del coach"
    return f"+{int(r.get('amount') or 0)} {que}"


def ficha(user_id):
    p = _perfil(user_id)
    if not p:
        return None
    pagado = p.get("plan_tier") or "gratis"
    es_admin = pagado == "admin"
    vigentes = [] if es_admin else rc.regalos_vigentes(user_id)
    cortesia = rc.cortesia_de(vigentes)
    efectivo = rc.plan_efectivo(pagado, cortesia) or "gratis"
    tope_plan, tope_coach = _topes(efectivo)
    extra_g, extra_c = rc.extra_de(vigentes, "generacion"), rc.extra_de(vigentes, "coach")
    historial = execute_sql_query(
        "SELECT id::text AS id, kind, amount, plan, starts_at, ends_at, reason, created_at, revoked_at, revoke_reason "
        "FROM public.account_grants WHERE user_id = %s ORDER BY created_at DESC LIMIT %s",
        (user_id, _HISTORIAL), fetch_all=True) or []
    ahora = datetime.now(timezone.utc)
    return {
        "user_id": p["id"], "email": p.get("email"), "nombre": p.get("full_name"), "alta": rc.iso(p.get("created_at")),
        "plan_pagado": pagado, "plan_efectivo": efectivo, "es_admin": es_admin,
        "suscripcion": {"estado": p.get("subscription_status"), "fin": rc.iso(p.get("subscription_end_date")),
                        "paypal": bool(p.get("tiene_paypal"))},
        "cortesia": ({"plan": cortesia["plan"], "hasta": rc.iso(cortesia["hasta"])}
                     if cortesia and efectivo == cortesia["plan"] and efectivo != pagado else None),
        "creditos": {"usados": int(get_monthly_api_usage(user_id) or 0), "plan": tope_plan, "regalo": extra_g,
                     "tope": tope_plan + extra_g},
        "coach": {"usados": int(get_monthly_api_usage(user_id, kind="coach") or 0), "plan": tope_coach,
                  "regalo": extra_c, "tope": tope_coach + extra_c},
        "regalos": [{"id": r["id"], "tipo": r.get("kind"), "detalle": _detalle(r), "desde": rc.iso(r.get("starts_at")),
                     "hasta": rc.iso(r.get("ends_at")), "motivo": r.get("reason"), "estado": _estado(r, ahora),
                     "motivo_reversion": r.get("revoke_reason")} for r in historial],
        "validez_creditos": {"mes": rc.iso(rc.inicio_de_mes(1)), "mes_siguiente": rc.iso(rc.inicio_de_mes(2))},
    }


def _motivo(v) -> str:
    m = " ".join(str(v or "").split())
    if not 3 <= len(m) <= 300:
        raise ErrorRegalo(422, "El motivo es obligatorio (de 3 a 300 caracteres).")
    return m


def _exigir_activo() -> None:
    if not rc.activo():
        raise ErrorRegalo(503, "Los regalos están apagados (MEALFIT_ACCOUNT_GRANTS).")


def _regalable(user_id) -> dict:
    p = _perfil(user_id)
    if not p:
        raise ErrorRegalo(404, "No existe esa cuenta.")
    if (p.get("plan_tier") or "") == "admin":
        raise ErrorRegalo(409, "Es una cuenta de administración: no se le regala nada.")
    return p


def _anotar(admin_id, accion, user_id, detalle) -> None:
    try:
        registrar_acceso(admin_id, accion, user_id, detalle)
    except Exception as e:  # noqa: BLE001
        logger.error(f"🛑 [P1-PLAN-LOTE-774] rastro no anotado ({accion} → {user_id}): {e!r}")
        raise ErrorRegalo(503, "No se pudo registrar la acción; no se hizo ningún cambio.") from e


def _anotar_fallo(admin_id, accion, user_id, detalle) -> None:
    """El rastro ya dice `accion` (se anota ANTES de escribir); si el cambio no se hizo, deja `accion_fallo` al lado.
    Best-effort: el fallo ya está en el log."""
    try:
        registrar_acceso(admin_id, f"{accion}_fallo", user_id, detalle)
    except Exception:  # noqa: BLE001
        pass


def _fallo_al_guardar(admin_id, accion, user_id, grant_id, e) -> ErrorRegalo:
    logger.error(f"🛑 [P1-PLAN-LOTE-774] {accion} no guardado para {user_id}: {e!r}")
    _anotar_fallo(admin_id, accion, user_id, {"grant_id": grant_id, "error": type(e).__name__})
    if type(e).__name__ == "UniqueViolation":
        return ErrorRegalo(409, "Se acaba de dar otra cortesía a esta cuenta; recarga la ficha.")
    return ErrorRegalo(500, "No se pudo guardar el regalo.")


def _invalidar_plan(user_id) -> None:
    try:
        from llm_provider import invalidate_tier_cache
        invalidate_tier_cache(user_id)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"⚠️ [P1-PLAN-LOTE-774] caché del plan no invalidada para {user_id}: {e!r}")


def regalar_creditos(admin_id, user_id, medidor, modo, cantidad, hasta, motivo) -> dict:
    _exigir_activo()
    if medidor not in rc.MEDIDORES:
        raise ErrorRegalo(422, "Medidor desconocido.")
    if hasta not in ("mes", "mes_siguiente"):
        raise ErrorRegalo(422, "Validez desconocida.")
    motivo = _motivo(motivo)
    _regalable(user_id)
    antes = ficha(user_id)["creditos" if medidor == "generacion" else "coach"]
    if modo == "completo":
        # El cupo del plan entero otra vez: lo gastado del plan, sin contar lo que ya cubre un regalo vigente.
        cantidad = min(antes["usados"] - antes["regalo"], MAX_CREDITOS)
        if cantidad < 1:
            raise ErrorRegalo(409, "Ya tiene disponible todo el cupo de su plan: no hay nada que recargar.")
    elif modo == "sumar":
        try:
            cantidad = int(cantidad)
        except (TypeError, ValueError):
            raise ErrorRegalo(422, "Cantidad inválida.") from None
        if not 1 <= cantidad <= MAX_CREDITOS:
            raise ErrorRegalo(422, f"La cantidad va de 1 a {MAX_CREDITOS}.")
    else:
        raise ErrorRegalo(422, "Modo desconocido.")
    fin = rc.inicio_de_mes(1 if hasta == "mes" else 2)
    gid = str(uuid.uuid4())
    kind = rc.MEDIDORES[medidor]
    _anotar(admin_id, "regalar_creditos", user_id, {
        "grant_id": gid, "medidor": medidor, "modo": modo, "cantidad": cantidad, "hasta": fin.isoformat(),
        "motivo": motivo, "antes": {"usados": antes["usados"], "tope": antes["tope"]},
        "despues": {"tope": antes["tope"] + cantidad}})
    try:
        execute_sql_write(
            "INSERT INTO public.account_grants (id, user_id, kind, amount, ends_at, reason, granted_by) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)", (gid, user_id, kind, cantidad, fin, motivo, admin_id))
    except Exception as e:  # noqa: BLE001
        raise _fallo_al_guardar(admin_id, "regalar_creditos", user_id, gid, e) from e
    avisar_en_segundo_plano(user_id, {"id": gid, "kind": kind, "amount": cantidad, "plan": None, "ends_at": fin})
    return {"grant_id": gid, "cantidad": cantidad, "hasta": fin.isoformat()}


def fin_de_cortesia(dia: date) -> datetime:
    """«Hasta el 31-oct» incluye todo el 31 en la hora de RD: termina el 1-nov a las 00:00 de Santo Domingo."""
    return datetime.combine(dia + timedelta(days=1), time(0), tzinfo=_ZONA_RD)


def dar_cortesia(admin_id, user_id, plan, hasta, motivo) -> dict:
    _exigir_activo()
    if plan not in rc.PLANES_REGALABLES:
        raise ErrorRegalo(422, "Ese plan no se puede regalar.")
    motivo = _motivo(motivo)
    pagado = _regalable(user_id).get("plan_tier") or "gratis"
    if rc.RANGO.get(plan, 0) <= rc.RANGO.get(pagado, 0):
        raise ErrorRegalo(409, f"Ya paga {_NOMBRE_PLAN.get(pagado, pagado)}: la cortesía tiene que ser un plan mejor.")
    fin = None
    if hasta:
        try:
            dia = date.fromisoformat(str(hasta)[:10])
        except ValueError:
            raise ErrorRegalo(422, "Fecha inválida.") from None
        hoy = datetime.now(_ZONA_RD).date()
        if dia < hoy or dia > hoy + timedelta(days=MAX_DIAS_CORTESIA):
            raise ErrorRegalo(422, "La fecha tiene que estar entre hoy y dentro de un año.")
        fin = fin_de_cortesia(dia)
    antes = ficha(user_id)
    gid = str(uuid.uuid4())
    _anotar(admin_id, "dar_cortesia", user_id, {
        "grant_id": gid, "plan": plan, "hasta": rc.iso(fin), "motivo": motivo,
        "antes": {"plan_efectivo": antes["plan_efectivo"], "cortesia": antes["cortesia"]},
        "despues": {"plan_efectivo": plan}})
    try:
        # Una sola cortesía viva: la anterior se revoca en la MISMA transacción (el índice único corta la carrera).
        execute_sql_transaction([
            ("UPDATE public.account_grants SET revoked_at = now(), revoked_by = %s, revoke_reason = %s "
             "WHERE user_id = %s AND kind = 'plan' AND revoked_at IS NULL",
             (admin_id, "Reemplazada por una cortesía nueva", user_id)),
            ("INSERT INTO public.account_grants (id, user_id, kind, plan, ends_at, reason, granted_by) "
             "VALUES (%s, %s, 'plan', %s, %s, %s, %s)", (gid, user_id, plan, fin, motivo, admin_id)),
        ])
    except Exception as e:  # noqa: BLE001
        raise _fallo_al_guardar(admin_id, "dar_cortesia", user_id, gid, e) from e
    _invalidar_plan(user_id)
    avisar_en_segundo_plano(user_id, {"id": gid, "kind": "plan", "amount": None, "plan": plan, "ends_at": fin})
    return {"grant_id": gid, "plan": plan, "hasta": rc.iso(fin)}


def revocar(admin_id, grant_id, motivo) -> dict:
    """Revertir funciona aunque el knob esté apagado: quitar un regalo nunca es peligroso."""
    motivo = _motivo(motivo)
    try:
        gid = str(uuid.UUID(str(grant_id)))
    except ValueError:
        raise ErrorRegalo(404, "No existe ese regalo.") from None
    r = execute_sql_query("SELECT user_id::text AS user_id, kind, revoked_at FROM public.account_grants WHERE id = %s",
                          (gid,), fetch_one=True)
    if not r:
        raise ErrorRegalo(404, "No existe ese regalo.")
    if r.get("revoked_at"):
        raise ErrorRegalo(409, "Ese regalo ya estaba revertido.")
    _anotar(admin_id, "revocar_regalo", r["user_id"], {"grant_id": gid, "tipo": r.get("kind"), "motivo": motivo})
    try:
        filas = execute_sql_write(
            "UPDATE public.account_grants SET revoked_at = now(), revoked_by = %s, revoke_reason = %s "
            "WHERE id = %s AND revoked_at IS NULL RETURNING id", (admin_id, motivo, gid), returning=True)
    except Exception as e:  # noqa: BLE001
        raise _fallo_al_guardar(admin_id, "revocar_regalo", r["user_id"], gid, e) from e
    if not filas:
        # Carrera perdida: otro lo revirtió entre la lectura y el UPDATE. Este admin no revirtió nada.
        logger.warning(f"⚠️ [P1-PLAN-LOTE-774] revocar_regalo {gid}: ya estaba revertido (carrera perdida)")
        _anotar_fallo(admin_id, "revocar_regalo", r["user_id"], {"grant_id": gid, "error": "ya_revertido"})
        raise ErrorRegalo(409, "Ese regalo ya estaba revertido.")
    if r.get("kind") == "plan":
        _invalidar_plan(r["user_id"])
    return {"user_id": r["user_id"], "grant_id": gid}
