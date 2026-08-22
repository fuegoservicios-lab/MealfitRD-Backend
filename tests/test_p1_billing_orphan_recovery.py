"""[P1-BILLING-ORPHAN-RECOVERY · 2026-08-22] Un cobro cuyo `/verify` no llegó
deja de perderse en silencio.

LO QUE PASABA
    `POST /api/subscription/verify` es HOY el ÚNICO camino por el que una
    suscripción de PayPal se escribe en `user_profiles`. Lo dispara el navegador
    desde `onApprove`. Si entre la aprobación y esa llamada se cae la red, el
    usuario cierra la pestaña, o `/verify` devuelve 500/409, PayPal cobra y el
    sistema NO se entera:

      - `paypal_subscription_id` queda NULL para ese usuario, y
      - los webhooks `ACTIVATED` / `PAYMENT.SALE.COMPLETED` filtran por
        `WHERE paypal_subscription_id = %s` → 0 filas → no-op silencioso.

    Resultado: cobro recurrente sin acceso, sin alerta, y sin manera automática
    de atribuir esa suscripción a nadie (el `create` del frontend solo mandaba
    `plan_id`, así que la única pista era el email del comprador, a mano).

    Auditoría 2026-08-22: no existía cron de reconciliación. La pérdida era
    permanente.

EL FIX (dos mitades, y las DOS hacen falta)
    1. El frontend estampa `custom_id: <user_id>` al crear la suscripción, así
       que PayPal nos lo devuelve firmado dentro del webhook.
    2. El webhook `BILLING.SUBSCRIPTION.ACTIVATED` que no matchea ninguna fila
       adopta al huérfano usando ese `custom_id`.

    El tier SIGUE derivándose server-side del `plan_id` (I-Billing-1): el
    `custom_id` dice A QUIÉN, jamás QUÉ. Un `custom_id` que no mapea a un tier
    conocido no otorga nada.

LO QUE NO HACE, a propósito
    Si el perfil destino ya tiene OTRA suscripción viva, NO la pisa: cambiarla
    sin cancelar la vieja en PayPal es exactamente el doble cobro que
    P1-BILLING-UPGRADE-FAIL-LOUD cerró. Alerta y deja el caso al operador.

Tests FUNCIONALES sobre el handler real (mismo harness que
`test_p1_webhook_dedup_atomic.py`), no parser-based: lo que hay que probar es
que la fila se escribe, no que el texto exista.

VERIFICADOS POR MUTACIÓN (2026-08-22). La primera versión de estos tests pasaba
contra las TRES mutaciones — eran una coartada. Lo que faltaba:

  - El doble de BD no sabía que `user_profiles.id` es un `uuid`, así que
    `test_custom_id_no_uuid_no_revienta_el_webhook` no podía distinguir un
    backend que valida de uno que no. Ahora `_FakeDB._as_uuid` rechaza igual que
    Postgres.
  - `test_no_resucita_una_cancelada` miraba solo el status, y el guard de "¿la
    tiene ya alguien?" también gobierna la ALERTA: sin él, cada cancelación
    normal emitía un crítico falso. Por eso ahora asegura también el silencio.

Queda una mutación que sobrevive A PROPÓSITO: quitar el guard
`if profile.get("paypal_subscription_id")` no cambia la conducta, porque el
`AND (paypal_subscription_id IS NULL OR = %s)` del propio UPDATE la defiende
igual. Son dos capas de lo mismo (una da mejor mensaje, la otra cierra además la
carrera); el contrato observable — no pisar una sub viva — sí está anclado.
"""
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import routers.billing as billing


_USER = "11111111-2222-3333-4444-555555555555"
_OTHER_USER = "99999999-8888-7777-6666-555555555555"
_PLAN_PLUS = "P-PLUS-TEST"
_SUB = "I-NEWSUB"


class _FakeReq:
    def __init__(self, body_bytes, headers):
        self._body = body_bytes
        self.headers = headers

    async def body(self):
        return self._body


class _FakeDB:
    """`user_profiles` como dict id→fila + `app_kv_store` + captura de alertas."""

    def __init__(self, profiles=None):
        self.profiles = profiles if profiles is not None else {}
        self.kv = {}
        self.alerts = []

    # -- helpers de aserción -------------------------------------------------
    def alert_keys(self):
        return [a[0] for a in self.alerts]

    def _rows_by_sub(self, sub_id):
        return [p for p in self.profiles.values() if p.get("paypal_subscription_id") == sub_id]

    # -- doubles de db -------------------------------------------------------
    @staticmethod
    def _as_uuid(value):
        """Postgres rechaza un `id` no-UUID con `invalid input syntax for type uuid`.
        El doble DEBE hacerlo igual: sin esto, el test del `custom_id` basura pasa
        contra un backend que no valida nada (medido por mutación: sobrevivía)."""
        import uuid as _uuid
        try:
            _uuid.UUID(str(value))
        except (ValueError, AttributeError, TypeError):
            raise RuntimeError(f'invalid input syntax for type uuid: "{value}"')
        return value

    def execute_sql_query(self, query, params=None, fetch_one=False, fetch_all=False):
        q = " ".join(str(query).split())
        if "FROM public.user_profiles" in q or "FROM user_profiles" in q:
            if "paypal_subscription_id = %s" in q:
                rows = self._rows_by_sub(params[0])
            elif "id = %s" in q:
                key = self._as_uuid(params[0])
                rows = [self.profiles[key]] if key in self.profiles else []
            else:
                rows = list(self.profiles.values())
            if fetch_one:
                return rows[0] if rows else None
            return rows
        return [] if fetch_all else None

    def execute_sql_write(self, query, params=None, returning=False, lock_timeout_ms=None):
        q = " ".join(str(query).split())

        if "system_alerts" in q:
            self.alerts.append((params[0], params[1], params[2]))
            return True

        if q.startswith("INSERT INTO app_kv_store"):
            key = params[0]
            if key in self.kv:
                return []
            self.kv[key] = {"status": "processing"}
            return [{"key": key}]
        if q.startswith("SELECT 1 AS done FROM app_kv_store"):
            return [{"done": 1}] if self.kv.get(params[0], {}).get("status") == "done" else []
        if q.startswith("UPDATE app_kv_store"):
            self.kv[params[0]] = {"status": "done"}
            return True

        if "UPDATE public.user_profiles" in q:
            # SET y WHERE comparten el nombre `paypal_subscription_id`, así que
            # el troceo va por " WHERE " ANTES de leer placeholders — mezclarlos
            # consumía el discriminante como si fuera un valor del SET.
            head, _, where = q.partition(" WHERE ")
            p = list(params)
            sets, i = {}, 0
            for col in ("subscription_status", "plan_tier", "paypal_subscription_id"):
                if f"{col} = %s" in head:
                    sets[col] = p[i]
                    i += 1
            rest = p[i:]

            # `where.startswith`, NO `"id = %s" in where`: esa subcadena vive
            # DENTRO de `paypal_subscription_id = %s` y desviaba el UPDATE del
            # camino normal a la rama por-usuario.
            if where.startswith("id = %s"):
                target = self._as_uuid(rest[0])
                rows = [self.profiles[target]] if target in self.profiles else []
                # Guard opcional: no pisar otra sub viva.
                if rows and "paypal_subscription_id IS NULL OR paypal_subscription_id = %s" in where:
                    cur = rows[0].get("paypal_subscription_id")
                    if cur is not None and cur != rest[1]:
                        rows = []
            else:
                rows = self._rows_by_sub(rest[0])
                if "subscription_status <> 'CANCELLED'" in where:
                    rows = [r for r in rows if r.get("subscription_status") not in (None, "CANCELLED")]
                if "subscription_status = 'PAYMENT_RETRYING'" in where:
                    rows = [r for r in rows if r.get("subscription_status") == "PAYMENT_RETRYING"]

            for r in rows:
                r.update(sets)
            return [{"id": r["id"]} for r in rows] if returning else True

        return [] if returning else True


@pytest.fixture
def db(monkeypatch):
    fake = _FakeDB(
        profiles={
            _USER: {
                "id": _USER,
                "plan_tier": "free",
                "subscription_status": None,
                "paypal_subscription_id": None,
            }
        }
    )
    monkeypatch.setattr(billing, "execute_sql_write", fake.execute_sql_write)
    monkeypatch.setattr(billing, "execute_sql_query", fake.execute_sql_query)
    monkeypatch.setattr(billing, "is_production", lambda: False)
    monkeypatch.setenv("MEALFIT_ALLOW_WEBHOOK_UNSIGNED", "1")
    monkeypatch.setenv("PAYPAL_PLAN_PLUS_ID", _PLAN_PLUS)
    for k in ("PAYPAL_CLIENT_ID", "PAYPAL_SECRET", "PAYPAL_WEBHOOK_ID"):
        monkeypatch.delenv(k, raising=False)
    return fake


def _activated(tx_id, *, custom_id=_USER, plan_id=_PLAN_PLUS, sub_id=_SUB):
    resource = {"id": sub_id, "plan_id": plan_id}
    if custom_id is not None:
        resource["custom_id"] = custom_id
    body = json.dumps(
        {"event_type": "BILLING.SUBSCRIPTION.ACTIVATED", "resource": resource}
    ).encode("utf-8")
    return _FakeReq(body, {"paypal-transmission-id": tx_id})


def _call(req):
    return asyncio.run(billing.api_webhook_paypal(req, _rl=None))


# ---------------------------------------------------------------------------
# EL CORAZÓN: el cobro huérfano se recupera.
# ---------------------------------------------------------------------------
def test_activated_huerfano_se_atribuye_por_custom_id(db):
    """Nadie tiene esta sub (el /verify nunca llegó) → el webhook la adopta."""
    res = _call(_activated("TX-ORPHAN"))
    assert res == {"success": True}

    prof = db.profiles[_USER]
    assert prof["paypal_subscription_id"] == _SUB, (
        "El webhook ACTIVATED debe adoptar la suscripción huérfana usando "
        "`custom_id`; sin esto el usuario paga y nunca recibe el plan."
    )
    assert prof["plan_tier"] == "plus", (
        "El tier debe derivarse server-side del plan_id (I-Billing-1)."
    )
    assert prof["subscription_status"] == "ACTIVE"


def test_la_recuperacion_deja_rastro_en_system_alerts(db):
    """Otorgar un tier fuera del camino normal SIEMPRE se audita."""
    _call(_activated("TX-ALERT"))
    assert any(
        k.startswith("billing_orphan_subscription_recovered:") for k in db.alert_keys()
    ), (
        "La adopción de un huérfano debe emitir "
        "`billing_orphan_subscription_recovered:<user>:<sub>` — es un tier "
        f"concedido sin pasar por /verify. Alertas vistas: {db.alert_keys()}"
    )


# ---------------------------------------------------------------------------
# Los límites: qué NO debe otorgar.
# ---------------------------------------------------------------------------
def test_plan_desconocido_no_otorga_nada(db):
    """`custom_id` dice A QUIÉN, nunca QUÉ. Sin tier derivable no hay upgrade."""
    _call(_activated("TX-UNKNOWN-PLAN", plan_id="P-NO-MAPEADO"))

    prof = db.profiles[_USER]
    assert prof["plan_tier"] == "free"
    assert prof["paypal_subscription_id"] is None
    assert any(
        k.startswith("billing_orphan_subscription_unrecoverable:") for k in db.alert_keys()
    ), f"Debe alertar el huérfano no atribuible. Vistas: {db.alert_keys()}"


def test_sin_custom_id_no_inventa_dueno(db):
    """Suscripción vieja (creada antes del fix) → no hay a quién atribuirla."""
    _call(_activated("TX-NO-CUSTOM", custom_id=None))

    assert db.profiles[_USER]["paypal_subscription_id"] is None
    assert any(
        k.startswith("billing_orphan_subscription_unrecoverable:") for k in db.alert_keys()
    )


def test_custom_id_no_uuid_no_revienta_el_webhook(db):
    """Un `custom_id` basura no debe propagar `invalid input syntax for type uuid`
    (sería un 503 y 25 reintentos de PayPal contra un error determinista)."""
    res = _call(_activated("TX-JUNK", custom_id="no-soy-un-uuid"))
    assert res == {"success": True}
    assert db.profiles[_USER]["paypal_subscription_id"] is None


def test_perfil_inexistente_no_crea_filas(db):
    _call(_activated("TX-GHOST", custom_id=_OTHER_USER))
    assert _OTHER_USER not in db.profiles
    assert any(
        k.startswith("billing_orphan_subscription_unrecoverable:") for k in db.alert_keys()
    )


def test_no_pisa_una_suscripcion_viva_del_mismo_usuario(db):
    """Reemplazar el handle sin cancelar la vieja en PayPal es el doble cobro
    que P1-BILLING-UPGRADE-FAIL-LOUD cerró. Alerta y NO toca."""
    db.profiles[_USER].update(
        {"paypal_subscription_id": "I-VIEJA", "subscription_status": "ACTIVE", "plan_tier": "basic"}
    )

    _call(_activated("TX-CLASH"))

    prof = db.profiles[_USER]
    assert prof["paypal_subscription_id"] == "I-VIEJA", (
        "NO debe pisar una suscripción viva: la vieja seguiría cobrando en PayPal "
        "sin handle para cancelarla."
    )
    assert prof["plan_tier"] == "basic"
    assert any(
        k.startswith("billing_orphan_subscription_unrecoverable:") for k in db.alert_keys()
    )


def test_el_camino_normal_no_se_toca(db):
    """Si la sub YA está en el perfil (el /verify sí llegó), la rama de
    recuperación no debe activarse ni alertar."""
    db.profiles[_USER].update(
        {"paypal_subscription_id": _SUB, "subscription_status": "PAYMENT_RETRYING", "plan_tier": "plus"}
    )

    _call(_activated("TX-NORMAL"))

    prof = db.profiles[_USER]
    assert prof["subscription_status"] == "ACTIVE"
    assert prof["plan_tier"] == "plus"
    assert not any("orphan" in k for k in db.alert_keys()), (
        f"El camino normal no debe emitir alertas de huérfano. Vistas: {db.alert_keys()}"
    )


def test_no_resucita_una_cancelada(db):
    """El guard P1-BILLING-REACTIVATE-NOT-CANCELLED manda: una sub CANCELLED por
    el usuario no es un huérfano, y la recuperación NO puede ser la puerta
    trasera que lo reviva."""
    db.profiles[_USER].update(
        {"paypal_subscription_id": _SUB, "subscription_status": "CANCELLED", "plan_tier": "plus"}
    )

    _call(_activated("TX-CANCELLED"))

    assert db.profiles[_USER]["subscription_status"] == "CANCELLED", (
        "Una sub CANCELLED debe seguir cancelada — reactivarla por la vía de "
        "recuperación da acceso pagado perpetuo sin cobro."
    )
    # Y tampoco debe ALERTAR: la sub tiene dueño, el UPDATE no matchó por el guard
    # de P1-BILLING-REACTIVATE-NOT-CANCELLED, no por orfandad. Sin esta aserción,
    # borrar el early-return de "¿la tiene ya alguien?" pasaba desapercibido y cada
    # cancelación normal emitía un crítico falso (medido por mutación).
    assert not any("orphan" in k for k in db.alert_keys()), (
        f"Una sub con dueño no es huérfana: no debe alertar. Vistas: {db.alert_keys()}"
    )


# ---------------------------------------------------------------------------
# Kill-switch.
# ---------------------------------------------------------------------------
def test_knob_apagado_revierte_a_la_conducta_previa(db, monkeypatch):
    monkeypatch.setenv("MEALFIT_BILLING_ORPHAN_RECOVERY", "false")

    _call(_activated("TX-OFF"))

    assert db.profiles[_USER]["paypal_subscription_id"] is None, (
        "Con el knob en false la recuperación no corre (conducta previa al fix)."
    )
