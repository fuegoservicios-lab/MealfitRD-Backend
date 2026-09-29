"""[P1-PLAN-LOTE-714 · 2026-09-28] La tarjeta «Suscripción» de Configuración, lista para producción.

Cinco defectos del backend de cobro (`routers/billing.py`), probados por CONDUCTA sobre los handlers reales —
PayPal y la base sustituidos por dobles en memoria; jamás la red ni Neon:

  1. `POST /api/subscription/cancel` exigía un body con `user_id` (sin body: 422). Ahora el usuario es el del JWT; el
     body es opcional y, si trae OTRO `user_id`, 401 como antes. Tiene limitador (5/min); un timeout o fallo de red
     con PayPal es un 502 con alerta `billing_cancel_failed` (antes un 500 mudo); y responde `access_until`.
  2. Cancelar sin `next_billing_time` escribía solo el estado, y el cargador del perfil degradaba AL INSTANTE a quien
     había pagado el mes. Ahora el fin sale del último pago + un intervalo del plan (mensual o anual). La prueba usa
     el consumidor REAL, `db_profiles.get_user_profile`, para comprobar que no degrada.
  3. La recuperación de huérfanos adoptaba sin la verificación de monto de `/verify` (el precio sobrescrito a $0.01
     entraba por la puerta de atrás) y rechazaba al ex-suscriptor cuya sub vieja ya estaba CANCELLED.
  4. `subscription_end_date` no se limpiaba al volver a suscribirse: Configuración pintaba un «Próximo cobro» falso.
  5. `GET /api/subscription/status`: forma exacta, caché, fail-soft si PayPal cae, sin payload crudo ni paywall.

El doble de base es un intérprete ESTRICTO de las sentencias que el router emite: una sentencia o condición que no
reconoce lanza, en vez de «pasar» en silencio. Así un cambio de SQL no deja estos tests verdes por accidente.

tooltip-anchor: P1-PLAN-LOTE-714
"""
import asyncio
import calendar
import json
import os
import re
import sys
import types
import uuid
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx  # noqa: E402
import pytest  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import rate_limiter  # noqa: E402
import routers.billing as billing  # noqa: E402
from auth import get_verified_user_id, verify_api_quota  # noqa: E402
from rate_limiter import RateLimiter  # noqa: E402

_USER = "11111111-2222-3333-4444-555555555555"
_OTHER = "99999999-8888-7777-6666-555555555555"
_SUB = "I-SUB-714"
_PLAN_M = "P-PLUS-MENSUAL-714"
_PLAN_Y = "P-PLUS-ANUAL-714"
_CANCELLED = "BILLING.SUBSCRIPTION.CANCELLED"
_ACTIVATED = "BILLING.SUBSCRIPTION.ACTIVATED"


# ---------------------------------------------------------------------------
# Utilidades de tiempo
# ---------------------------------------------------------------------------
def _ahora() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _iso_paypal(dt: datetime) -> str:
    """Como las escribe PayPal: `2026-10-01T10:00:00Z`."""
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _mas_meses(dt: datetime, n: int) -> datetime:
    y, m = divmod(dt.month - 1 + n, 12)
    y += dt.year
    m += 1
    return dt.replace(year=y, month=m, day=min(dt.day, calendar.monthrange(y, m)[1]))


def _parse_ts(value):
    if value is None or isinstance(value, datetime):
        return value
    dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Doble de base: `user_profiles` + `discount_codes` + `app_kv_store`
# ---------------------------------------------------------------------------
class _FakeDB:
    def __init__(self):
        self.profiles = {}
        self.coupons = []  # filas activas de discount_codes
        self.kv = {}

    def add_profile(self, uid=_USER, **cols):
        fila = {
            "id": uid, "plan_tier": "gratis", "subscription_status": None,
            "paypal_subscription_id": None, "subscription_end_date": None,
        }
        fila.update(cols)
        self.profiles[uid] = fila
        return fila

    @staticmethod
    def _uuid(value):
        """Postgres rechaza un `id` no-UUID; el doble también."""
        try:
            uuid.UUID(str(value))
        except (ValueError, AttributeError, TypeError):
            raise RuntimeError(f'invalid input syntax for type uuid: "{value}"')
        return str(value)

    # -- lecturas -----------------------------------------------------------
    def execute_sql_query(self, query, params=None, fetch_one=False, fetch_all=False):
        q = " ".join(str(query).split())
        if "FROM public.discount_codes" in q:
            filas = [dict(c) for c in self.coupons]
        elif re.search(r"FROM (public\.)?user_profiles WHERE ", q):
            where = q.split(" WHERE ", 1)[1]
            if where.startswith("paypal_subscription_id = %s"):
                filas = [dict(r) for r in self.profiles.values() if r["paypal_subscription_id"] == params[0]]
            elif where.startswith("id = %s"):
                fila = self.profiles.get(self._uuid(params[0]))
                filas = [dict(fila)] if fila else []
            else:
                raise AssertionError(f"SELECT no soportado por el doble: {q}")
        else:
            raise AssertionError(f"SELECT no soportado por el doble: {q}")
        if fetch_one:
            return filas[0] if filas else None
        return filas

    # -- escrituras ---------------------------------------------------------
    def execute_sql_write(self, query, params=None, returning=False, lock_timeout_ms=None):
        q = " ".join(str(query).split())
        if q.startswith("INSERT INTO app_kv_store"):
            if params[0] in self.kv:
                return []
            self.kv[params[0]] = "processing"
            return [{"key": params[0]}]
        if q.startswith("SELECT 1 AS done FROM app_kv_store"):
            return [{"done": 1}] if self.kv.get(params[0]) == "done" else []
        if q.startswith("UPDATE app_kv_store"):
            self.kv[params[0]] = "done"
            return True
        if q.startswith("UPDATE user_profiles SET plan_tier = %(plan_tier)s"):
            # La degradación de `db_profiles.get_user_profile` (placeholders con nombre).
            fila = self.profiles.get(params["id"])
            if fila:
                fila.update(plan_tier=params["plan_tier"], subscription_status=params["subscription_status"])
            return True
        if q.startswith("UPDATE public.user_profiles SET "):
            return self._update_profiles(q, list(params or ()), returning)
        raise AssertionError(f"escritura no soportada por el doble: {q}")

    @staticmethod
    def _partir(texto, separador):
        """Parte por `separador` solo fuera de paréntesis y de bloques CASE … END."""
        partes, prof, caso, buf, i = [], 0, 0, "", 0
        while i < len(texto):
            if re.match(r"\bCASE\b", texto[i:]) and (i == 0 or not texto[i - 1].isalnum()):
                caso += 1
            elif re.match(r"\bEND\b", texto[i:]) and (i == 0 or not texto[i - 1].isalnum()):
                caso -= 1
            c = texto[i]
            if c == "(":
                prof += 1
            elif c == ")":
                prof -= 1
            if prof == 0 and caso == 0 and texto.startswith(separador, i):
                partes.append(buf.strip())
                buf, i = "", i + len(separador)
                continue
            buf += c
            i += 1
        if buf.strip():
            partes.append(buf.strip())
        return partes

    def _update_profiles(self, q, params, returning):
        cabeza, _, resto = q.partition(" WHERE ")
        where, _, cols_ret = resto.partition(" RETURNING ")
        i = 0
        asignaciones = []
        for asig in self._partir(cabeza.split(" SET ", 1)[1], ","):
            col, _, expr = asig.partition(" = ")
            if expr == "%s":
                asignaciones.append((col, "valor", params[i]))
                i += 1
            elif expr == "%s::timestamptz":
                asignaciones.append((col, "valor", _parse_ts(params[i])))
                i += 1
            elif expr == "NULL":
                asignaciones.append((col, "valor", None))
            elif re.fullmatch(r"'[^']*'", expr):
                asignaciones.append((col, "valor", expr[1:-1]))
            elif expr == ("CASE WHEN subscription_status = 'CANCELLED' AND subscription_end_date IS NOT NULL "
                          "THEN LEAST(subscription_end_date, %s::timestamptz) ELSE %s::timestamptz END"):
                asignaciones.append((col, "least_si_cancelada", (_parse_ts(params[i]), _parse_ts(params[i + 1]))))
                i += 2
            else:
                raise AssertionError(f"asignación no soportada por el doble: {asig!r}")

        predicados = []
        for cond in self._partir(where, " AND "):
            if cond == "id = %s":
                v = self._uuid(params[i])
                i += 1
                predicados.append(lambda r, v=v: r["id"] == v)
            elif cond == "paypal_subscription_id = %s":
                v = params[i]
                i += 1
                predicados.append(lambda r, v=v: r["paypal_subscription_id"] == v)
            elif cond == "subscription_status <> 'CANCELLED'":
                # NULL <> 'CANCELLED' es NULL en SQL: no matchea.
                predicados.append(lambda r: r["subscription_status"] not in (None, "CANCELLED"))
            elif cond == "subscription_status = 'PAYMENT_RETRYING'":
                predicados.append(lambda r: r["subscription_status"] == "PAYMENT_RETRYING")
            elif cond == "(paypal_subscription_id IS NULL OR paypal_subscription_id = %s)":
                # La forma del guard ANTERIOR a este lote: se soporta para que, si alguien la restaura, estos tests
                # fallen por CONDUCTA (qué se adoptó) y no porque el doble se niegue a interpretarla.
                v = params[i]
                i += 1
                predicados.append(lambda r, v=v: r["paypal_subscription_id"] in (None, v))
            elif cond.startswith("(paypal_subscription_id IS NULL OR paypal_subscription_id = %s OR "
                                 "upper(COALESCE(subscription_status, '')) NOT IN ("):
                v = params[i]
                i += 1
                vivos = set(re.findall(r"'([A-Z_]+)'", cond.split(" NOT IN ", 1)[1]))
                predicados.append(
                    lambda r, v=v, vivos=vivos: r["paypal_subscription_id"] is None
                    or r["paypal_subscription_id"] == v
                    or (r["subscription_status"] or "").upper() not in vivos
                )
            else:
                raise AssertionError(f"condición WHERE no soportada por el doble: {cond!r}")
        assert i == len(params), f"parámetros sin consumir: {params[i:]} en {q}"

        tocadas = []
        for fila in self.profiles.values():
            if not all(p(fila) for p in predicados):
                continue
            vieja = dict(fila)  # SQL evalúa todo el SET contra la fila VIEJA
            for col, tipo, valor in asignaciones:
                if tipo == "valor":
                    fila[col] = valor
                else:
                    menor, sino = valor
                    actual = vieja["subscription_end_date"]
                    if vieja["subscription_status"] == "CANCELLED" and actual is not None:
                        fila[col] = min(actual, menor)
                    else:
                        fila[col] = sino
            tocadas.append(fila)
        if not returning:
            return True
        cols = [c.strip() for c in cols_ret.split(",")] if cols_ret else ["id"]
        return [{c: f[c] for c in cols} for f in tocadas]


# ---------------------------------------------------------------------------
# Doble de PayPal (sustituye a `httpx.AsyncClient` solo dentro de routers.billing)
# ---------------------------------------------------------------------------
class _FakePayPal:
    def __init__(self):
        self.subs = {}           # sub_id -> recurso que devuelve el GET
        self.plans = {}          # plan_id -> recurso del GET del plan
        self.token_status = 200
        self.cancel_status = 204
        self.cancel_json = None
        self.fail = {}           # etapa -> excepción httpx: oauth | get | cancel | plan | verify
        self.get_overrides = []  # códigos de estado que consumen los próximos GET de la sub, en orden
        self.retraso_s = 0.0     # PayPal «colgado»: cada llamada tarda esto
        self.calls = []          # (método, ruta)

    def client_class(self):
        fake = self

        class _Client:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def post(self, url, **kwargs):
                if fake.retraso_s:
                    await asyncio.sleep(fake.retraso_s)
                return fake._responder("POST", url)

            async def get(self, url, **kwargs):
                if fake.retraso_s:
                    await asyncio.sleep(fake.retraso_s)
                return fake._responder("GET", url)

        return _Client

    def _responder(self, metodo, url):
        ruta = url.split("paypal.com", 1)[1]
        self.calls.append((metodo, ruta))
        if ruta == "/v1/oauth2/token":
            etapa = "oauth"
        elif ruta == "/v1/notifications/verify-webhook-signature":
            etapa = "verify"
        elif ruta.startswith("/v1/billing/plans/"):
            etapa = "plan"
        elif ruta.endswith("/cancel"):
            etapa = "cancel"
        elif ruta.startswith("/v1/billing/subscriptions/"):
            etapa = "get"
        else:
            raise AssertionError(f"llamada a PayPal inesperada: {metodo} {url}")
        if etapa in self.fail:
            raise self.fail[etapa]
        if etapa == "oauth":
            return httpx.Response(self.token_status, json={"access_token": "tok-714-secreto"})
        if etapa == "verify":
            return httpx.Response(200, json={"verification_status": "SUCCESS"})
        if etapa == "plan":
            plan = self.plans.get(ruta.rsplit("/", 1)[1])
            return httpx.Response(200, json=plan) if plan else httpx.Response(404, json={})
        if etapa == "cancel":
            sub = self.subs.get(ruta.split("/")[-2])
            if sub is not None and self.cancel_status in (200, 204):
                # Como PayPal: una sub cancelada ya no tiene próximo cobro.
                sub["status"] = "CANCELLED"
                sub.get("billing_info", {}).pop("next_billing_time", None)
            if self.cancel_json is not None:
                return httpx.Response(self.cancel_status, json=self.cancel_json)
            return httpx.Response(self.cancel_status)
        if self.get_overrides:
            return httpx.Response(self.get_overrides.pop(0), json={"name": "INTERNAL_SERVER_ERROR"})
        sub = self.subs.get(ruta.rsplit("/", 1)[1])
        if sub is None:
            return httpx.Response(404, json={"name": "RESOURCE_NOT_FOUND"})
        return httpx.Response(200, json=sub)

    def llamo(self, metodo, ruta):
        return (metodo, ruta) in self.calls


class _HttpxShim(types.ModuleType):
    """`httpx` para routers.billing: el AsyncClient es el doble; todo lo demás (excepciones) es el real."""

    def __init__(self, client_cls):
        super().__init__("httpx_p1_plan_lote_714")
        self.AsyncClient = client_cls

    def __getattr__(self, name):
        return getattr(httpx, name)


class _Entorno:
    def __init__(self, db, paypal, alertas):
        self.db = db
        self.paypal = paypal
        self.alertas = alertas
        self.uid_jwt = _USER

    def claves(self):
        return [a["alert_key"] for a in self.alertas]

    def alertas_de(self, clave):
        return [a for a in self.alertas if a["alert_key"] == clave]


_KNOBS_A_LIMPIAR = (
    "MEALFIT_BILLING_VERIFY_AMOUNT", "MEALFIT_BILLING_AMOUNT_TOLERANCE_PCT", "MEALFIT_BILLING_ORPHAN_RECOVERY",
    "MEALFIT_ALLOW_WEBHOOK_UNSIGNED", "MEALFIT_ALLOW_PAYPAL_BYPASS", "MEALFIT_BILLING_REACTIVATE_NOT_CANCELLED",
    "MEALFIT_BILLING_PAYMENT_FAILED_GRACE",
)


def _limpiar_estado_de_proceso():
    for valor in vars(billing).values():
        if isinstance(valor, RateLimiter):
            valor._hits.clear()
    cache = getattr(billing, "_PAYPAL_STATUS_CACHE", None)
    if isinstance(cache, dict):
        cache.clear()


@pytest.fixture
def entorno(monkeypatch):
    db, paypal, alertas = _FakeDB(), _FakePayPal(), []
    monkeypatch.setattr(billing, "execute_sql_query", db.execute_sql_query)
    monkeypatch.setattr(billing, "execute_sql_write", db.execute_sql_write)
    monkeypatch.setattr(billing, "_persist_billing_alert", lambda **kw: alertas.append(kw))
    monkeypatch.setattr(billing, "httpx", _HttpxShim(paypal.client_class()))
    monkeypatch.setattr(billing, "is_production", lambda: False)
    monkeypatch.setattr(rate_limiter, "redis_client", None)  # limitador en memoria, determinista
    for clave in [k for k in os.environ if k.startswith("PAYPAL_")] + list(_KNOBS_A_LIMPIAR):
        monkeypatch.delenv(clave, raising=False)
    monkeypatch.setenv("PAYPAL_CLIENT_ID", "cid-714")
    monkeypatch.setenv("PAYPAL_SECRET", "secreto-714")
    monkeypatch.setenv("PAYPAL_WEBHOOK_ID", "wh-714")
    monkeypatch.setenv("PAYPAL_PLAN_PLUS_ID", _PLAN_M)
    monkeypatch.setenv("PAYPAL_PLAN_PLUS_ANNUAL_ID", _PLAN_Y)
    _limpiar_estado_de_proceso()
    yield _Entorno(db, paypal, alertas)
    _limpiar_estado_de_proceso()


@pytest.fixture
def cliente(entorno):
    app = FastAPI()
    app.include_router(billing.router)
    app.dependency_overrides[get_verified_user_id] = lambda: entorno.uid_jwt
    with TestClient(app) as c:
        yield c


class _FakeReq:
    def __init__(self, body, headers):
        self._body = body
        self.headers = headers

    async def body(self):
        return self._body


def _webhook(evento, recurso, tx):
    """Webhook FIRMADO (PAYPAL_* configurado; el doble de PayPal verifica la firma)."""
    cuerpo = json.dumps({"event_type": evento, "resource": recurso}).encode("utf-8")
    return asyncio.run(billing.api_webhook_paypal(_FakeReq(cuerpo, {"paypal-transmission-id": tx}), _rl=None))


def _sub(sub_id=_SUB, plan_id=_PLAN_M, *, status="ACTIVE", nbt=None, pagado_en=None, monto="19.99", **extra):
    billing_info = {}
    if nbt is not None:
        billing_info["next_billing_time"] = _iso_paypal(nbt)
    if pagado_en is not None:
        billing_info["last_payment"] = {"amount": {"currency_code": "USD", "value": monto}, "time": _iso_paypal(pagado_en)}
    recurso = {"id": sub_id, "plan_id": plan_id, "status": status, "billing_info": billing_info}
    recurso.update(extra)
    return recurso


def _perfil_cargado(db, monkeypatch, uid=_USER):
    """El consumidor REAL de `subscription_end_date`: el cargador del perfil que degrada los CANCELLED vencidos."""
    import db_profiles
    monkeypatch.setattr(db_profiles, "connection_pool", object())
    monkeypatch.setattr(db_profiles, "execute_sql_query", db.execute_sql_query)
    monkeypatch.setattr(db_profiles, "execute_sql_write", db.execute_sql_write)
    return db_profiles.get_user_profile(uid)


def _perfil_de_pago(entorno, **cols):
    base = {"plan_tier": "plus", "subscription_status": "ACTIVE", "paypal_subscription_id": _SUB}
    base.update(cols)
    return entorno.db.add_profile(**base)


# ===========================================================================
# 1. POST /api/subscription/cancel
# ===========================================================================
def test_cancel_sin_body_cancela_la_suscripcion_del_usuario_del_jwt(entorno, cliente):
    """Antes: `data: dict = Body(...)` → sin body, 422 sin llegar al handler."""
    nbt = _ahora() + timedelta(days=20)
    _perfil_de_pago(entorno)
    entorno.paypal.subs[_SUB] = _sub(nbt=nbt, pagado_en=_ahora() - timedelta(days=10))

    r = cliente.post("/api/subscription/cancel")

    assert r.status_code == 200, r.text
    assert r.json()["success"] is True
    assert r.json()["access_until"] == nbt.isoformat(), "la UI necesita la fecha para «Acceso hasta …»"
    fila = entorno.db.profiles[_USER]
    assert fila["subscription_status"] == "CANCELLED"
    assert fila["subscription_end_date"] == nbt
    assert entorno.paypal.llamo("POST", f"/v1/billing/subscriptions/{_SUB}/cancel")


@pytest.mark.parametrize(
    "peticion",
    [
        {"headers": {"Content-Type": "application/json"}, "content": b""},
        {"json": {}},
        {"json": {"user_id": _USER}},
    ],
    ids=["content-type-json-cuerpo-vacio", "objeto-vacio", "user_id-propio"],
)
def test_cancel_con_content_type_json(entorno, cliente, peticion):
    nbt = _ahora() + timedelta(days=15)
    _perfil_de_pago(entorno)
    entorno.paypal.subs[_SUB] = _sub(nbt=nbt)

    r = cliente.post("/api/subscription/cancel", **peticion)

    assert r.status_code == 200, r.text
    assert r.json()["access_until"] == nbt.isoformat()
    assert entorno.db.profiles[_USER]["subscription_status"] == "CANCELLED"


def test_cancel_con_user_id_ajeno_en_el_body_es_401_y_no_toca_nada(entorno, cliente):
    _perfil_de_pago(entorno)
    entorno.paypal.subs[_SUB] = _sub(nbt=_ahora() + timedelta(days=15))

    r = cliente.post("/api/subscription/cancel", json={"user_id": _OTHER})

    assert r.status_code == 401
    assert entorno.paypal.calls == [], "un 401 no puede hablar con PayPal"
    assert entorno.db.profiles[_USER]["subscription_status"] == "ACTIVE"


def test_cancel_sin_sesion_es_401(entorno, cliente):
    entorno.uid_jwt = None
    _perfil_de_pago(entorno)

    r = cliente.post("/api/subscription/cancel")

    assert r.status_code == 401
    assert entorno.paypal.calls == []


@pytest.mark.parametrize(
    "etapa_fallida,excepcion,stage",
    [
        ("cancel", httpx.ReadTimeout("read timed out"), "cancel"),
        ("oauth", httpx.ConnectError("connection refused"), "oauth"),
        ("get", httpx.ConnectTimeout("connect timed out"), "subscription_get"),
    ],
    ids=["timeout-en-el-cancel", "red-caida-en-oauth", "timeout-en-el-get"],
)
def test_cancel_timeout_o_red_de_paypal_alerta_y_502(entorno, cliente, etapa_fallida, excepcion, stage):
    """Antes: la excepción de httpx caía al `except Exception` → 500 genérico, sin alerta (I-Billing-3)."""
    _perfil_de_pago(entorno)
    entorno.paypal.subs[_SUB] = _sub(nbt=_ahora() + timedelta(days=15))
    entorno.paypal.fail[etapa_fallida] = excepcion

    r = cliente.post("/api/subscription/cancel", json={"user_id": _USER})

    assert r.status_code == 502, r.text
    assert entorno.db.profiles[_USER]["subscription_status"] == "ACTIVE", "sin confirmación de PayPal la BD no se toca"
    alertas = entorno.alertas_de(f"billing_cancel_failed:{_USER}:{_SUB}")
    assert len(alertas) == 1, f"alertas vistas: {entorno.claves()}"
    assert alertas[0]["severity"] == "critical"
    assert alertas[0]["metadata"]["stage"] == stage
    assert alertas[0]["metadata"]["error"] == type(excepcion).__name__


def test_cancel_tiene_limitador_de_5_por_minuto(entorno, cliente):
    """Con `user_id` propio en el body para que el código viejo (sin limitador) llegue a responder 200."""
    _perfil_de_pago(entorno)
    entorno.paypal.subs[_SUB] = _sub(nbt=_ahora() + timedelta(days=15))

    codigos = [cliente.post("/api/subscription/cancel", json={"user_id": _USER}).status_code for _ in range(6)]

    assert codigos == [200] * 5 + [429], codigos


@pytest.mark.parametrize(
    "plan_id,dias_desde_el_pago,meses",
    [(_PLAN_M, 10, 1), (_PLAN_Y, 40, 12)],
    ids=["mensual", "anual"],
)
def test_cancel_sin_next_billing_time_calcula_el_fin_desde_el_ultimo_pago(
    entorno, cliente, monkeypatch, plan_id, dias_desde_el_pago, meses,
):
    """La sub ya estaba cancelada en PayPal (desde la cuenta de PayPal, sin que llegara el webhook): el GET no trae
    `next_billing_time` y el cancel responde 422 «ya cancelada» (idempotente). Antes: solo el estado → el cargador
    del perfil degradaba al instante. En el caso anual, «+1 mes» daría una fecha ya vencida: distingue el intervalo."""
    pagado_en = _ahora() - timedelta(days=dias_desde_el_pago)
    _perfil_de_pago(entorno)
    entorno.paypal.subs[_SUB] = _sub(plan_id=plan_id, status="CANCELLED", pagado_en=pagado_en)
    entorno.paypal.cancel_status = 422
    entorno.paypal.cancel_json = {"name": "UNPROCESSABLE_ENTITY", "details": [{"issue": "SUBSCRIPTION_STATUS_INVALID"}]}

    r = cliente.post("/api/subscription/cancel")

    esperado = _mas_meses(pagado_en, meses)
    assert r.status_code == 200, r.text
    assert r.json()["access_until"] == esperado.isoformat()
    fila = entorno.db.profiles[_USER]
    assert fila["subscription_status"] == "CANCELLED"
    assert fila["subscription_end_date"] == esperado
    perfil = _perfil_cargado(entorno.db, monkeypatch)
    assert perfil is not None and perfil["plan_tier"] == "plus", (
        "el período pagado sigue vigente: el cargador del perfil NO puede degradar todavía"
    )
    assert not [k for k in entorno.claves() if k.startswith("billing_cancel_access_end_unknown:")]


def test_cancel_sin_ningun_dato_de_pago_conserva_la_conducta_y_alerta(entorno, cliente, monkeypatch):
    """Ni `next_billing_time` ni último pago: se escribe solo el estado (degradar ya, fail-secure) y se alerta."""
    _perfil_de_pago(entorno)
    entorno.paypal.subs[_SUB] = _sub()  # billing_info vacío

    r = cliente.post("/api/subscription/cancel")

    assert r.status_code == 200, r.text
    assert r.json()["access_until"] is None
    fila = entorno.db.profiles[_USER]
    assert fila["subscription_status"] == "CANCELLED" and fila["subscription_end_date"] is None
    alertas = entorno.alertas_de(f"billing_cancel_access_end_unknown:{_USER}:{_SUB}")
    assert len(alertas) == 1, f"alertas vistas: {entorno.claves()}"
    assert alertas[0]["severity"] == "warning"
    assert _perfil_cargado(entorno.db, monkeypatch)["plan_tier"] == "gratis", "sin fecha no se concede gracia"


def test_cancel_reintenta_el_get_tras_cancelar_si_el_primero_fallo(entorno, cliente):
    """El GET previo al cancel respondió 500; tras cancelar se vuelve a pedir (el último pago sigue ahí)."""
    pagado_en = _ahora() - timedelta(days=3)
    _perfil_de_pago(entorno)
    entorno.paypal.subs[_SUB] = _sub(status="CANCELLED", pagado_en=pagado_en)
    entorno.paypal.get_overrides = [500]

    r = cliente.post("/api/subscription/cancel")

    assert r.status_code == 200, r.text
    assert r.json()["access_until"] == _mas_meses(pagado_en, 1).isoformat()


def test_cancelar_dos_veces_no_vuelve_a_llamar_a_paypal(entorno, cliente):
    fin = _ahora() + timedelta(days=12)
    _perfil_de_pago(entorno, subscription_status="CANCELLED", subscription_end_date=fin)

    r = cliente.post("/api/subscription/cancel", json={"user_id": _USER})

    assert r.status_code == 200, r.text
    assert r.json()["access_until"] == fin.isoformat()
    assert entorno.paypal.calls == []
    assert entorno.db.profiles[_USER]["subscription_end_date"] == fin


# ===========================================================================
# 2. Webhook BILLING.SUBSCRIPTION.CANCELLED sin next_billing_time
# ===========================================================================
@pytest.mark.parametrize(
    "plan_id,dias_desde_el_pago,meses",
    [(_PLAN_M, 10, 1), (_PLAN_Y, 40, 12)],
    ids=["mensual", "anual"],
)
def test_webhook_cancelled_usa_el_ultimo_pago_del_recurso(entorno, monkeypatch, plan_id, dias_desde_el_pago, meses):
    """Cancelación desde la cuenta de PayPal: el recurso firmado trae el último pago pero no `next_billing_time`."""
    pagado_en = _ahora() - timedelta(days=dias_desde_el_pago)
    _perfil_de_pago(entorno)

    res = _webhook(_CANCELLED, _sub(plan_id=plan_id, status="CANCELLED", pagado_en=pagado_en), "TX-714-C1")

    assert res == {"success": True}
    fila = entorno.db.profiles[_USER]
    assert fila["subscription_status"] == "CANCELLED"
    assert fila["subscription_end_date"] == _mas_meses(pagado_en, meses)
    assert _perfil_cargado(entorno.db, monkeypatch)["plan_tier"] == "plus"
    assert not [k for k in entorno.claves() if "access_end_unknown" in k]


def test_webhook_cancelled_sin_billing_info_pregunta_a_paypal(entorno, monkeypatch):
    pagado_en = _ahora() - timedelta(days=5)
    _perfil_de_pago(entorno)
    entorno.paypal.subs[_SUB] = _sub(status="CANCELLED", pagado_en=pagado_en)

    _webhook(_CANCELLED, {"id": _SUB, "status": "CANCELLED"}, "TX-714-C2")

    assert entorno.paypal.llamo("GET", f"/v1/billing/subscriptions/{_SUB}")
    assert entorno.db.profiles[_USER]["subscription_end_date"] == _mas_meses(pagado_en, 1)
    assert _perfil_cargado(entorno.db, monkeypatch)["plan_tier"] == "plus"


def test_webhook_cancelled_indeterminado_alerta_y_conserva_la_conducta(entorno):
    _perfil_de_pago(entorno)
    entorno.paypal.fail["get"] = httpx.ReadTimeout("read timed out")

    res = _webhook(_CANCELLED, {"id": _SUB, "plan_id": _PLAN_M, "status": "CANCELLED", "billing_info": {}}, "TX-714-C3")

    assert res == {"success": True}, "el estado CANCELLED se registra igual: perderlo dejaría el tier pagado sin cobro"
    fila = entorno.db.profiles[_USER]
    assert fila["subscription_status"] == "CANCELLED" and fila["subscription_end_date"] is None
    alertas = entorno.alertas_de(f"billing_cancel_access_end_unknown:{_USER}:{_SUB}")
    assert len(alertas) == 1 and alertas[0]["severity"] == "warning", f"alertas vistas: {entorno.claves()}"


def test_webhook_cancelled_no_alarga_la_fecha_que_escribio_cancel(entorno):
    """`/cancel` ya escribió el `next_billing_time` de PayPal; el webhook calcula «último pago + 1 mes», que se pasa
    unas horas (el cobro se captura después del inicio del ciclo). Se queda la MENOR: nunca más de lo pagado."""
    base = _ahora() + timedelta(days=20)
    nbt = base.replace(day=min(base.day, 28))
    pagado_en = _mas_meses(nbt, -1) + timedelta(hours=3)
    _perfil_de_pago(entorno, subscription_status="CANCELLED", subscription_end_date=nbt)

    _webhook(_CANCELLED, _sub(status="CANCELLED", pagado_en=pagado_en), "TX-714-C4")

    assert entorno.db.profiles[_USER]["subscription_end_date"] == nbt
    assert not [k for k in entorno.claves() if "access_end_unknown" in k]


# ===========================================================================
# 3. Recuperación de huérfanos: verificación de monto + ex-suscriptor
# ===========================================================================
def _huerfano_sobrescrito(entorno, monto="0.01"):
    """Sub ACTIVE que nadie tiene, con `plan_overridden` y un primer cobro por debajo del precio de lista (19.99)."""
    sub = _sub(nbt=_ahora() + timedelta(days=30), pagado_en=_ahora(), monto=monto,
               plan_overridden=True, custom_id=_USER)
    entorno.paypal.subs[_SUB] = sub
    entorno.paypal.plans[_PLAN_M] = {
        "id": _PLAN_M,
        "billing_cycles": [{"tenure_type": "REGULAR",
                            "pricing_scheme": {"fixed_price": {"value": "19.99", "currency_code": "USD"}}}],
    }
    return sub


def test_huerfano_con_precio_sobrescrito_y_sin_cupon_posible_no_se_adopta(entorno):
    """El ataque de la auditoría (§1): precio sobrescrito desde la consola. `/verify` lo rechaza con 409; antes el
    webhook ACTIVATED lo adoptaba igual por el `custom_id`. Ahora corre la MISMA verificación y no adopta."""
    entorno.db.add_profile(plan_tier="gratis")
    sub = _huerfano_sobrescrito(entorno)

    res = _webhook(_ACTIVATED, sub, "TX-714-O1")

    assert res == {"success": True}
    fila = entorno.db.profiles[_USER]
    assert fila["paypal_subscription_id"] is None and fila["plan_tier"] == "gratis", "un cobro de $0.01 no da Plus"
    tampering = entorno.alertas_de(f"billing_price_tampering:{_USER}:{_SUB}")
    no_adoptada = entorno.alertas_de(f"billing_orphan_subscription_unrecoverable:{_SUB}")
    assert tampering and tampering[0]["severity"] == "critical", f"alertas vistas: {entorno.claves()}"
    assert no_adoptada and no_adoptada[0]["severity"] == "critical", f"alertas vistas: {entorno.claves()}"
    assert "monto" in no_adoptada[0]["metadata"]["motivo"]
    assert not [k for k in entorno.claves() if k.startswith("billing_orphan_subscription_recovered:")]


def test_huerfano_con_override_y_un_cupon_posible_se_adopta_con_alerta(entorno):
    """Misma política que `/verify`: si existe un cupón que pudiera explicar el override (el cliente no lo reenvió),
    es ambiguo — se alerta y NO se bloquea un pago que no se puede probar fraudulento."""
    entorno.db.add_profile(plan_tier="gratis")
    entorno.db.coupons = [{"applicable_tiers": ["plus"], "valid_from": None, "valid_until": None}]
    sub = _huerfano_sobrescrito(entorno, monto="9.99")

    _webhook(_ACTIVATED, sub, "TX-714-O2")

    fila = entorno.db.profiles[_USER]
    assert fila["paypal_subscription_id"] == _SUB and fila["plan_tier"] == "plus"
    assert entorno.alertas_de(f"billing_price_tampering:{_USER}:{_SUB}"), f"alertas vistas: {entorno.claves()}"
    assert entorno.alertas_de(f"billing_orphan_subscription_recovered:{_USER}:{_SUB}")


@pytest.mark.parametrize("estado_viejo", ["CANCELLED", "INACTIVE"])
def test_ex_suscriptor_cuya_sub_vieja_ya_no_cobra_se_adopta(entorno, estado_viejo):
    """Volvió, pagó y cerró la pestaña. Antes: cualquier `paypal_subscription_id` en el perfil bloqueaba la adopción
    y se quedaba en gratis mientras PayPal le cobraba."""
    entorno.db.add_profile(plan_tier="gratis", subscription_status=estado_viejo,
                           paypal_subscription_id="I-VIEJA", subscription_end_date=_ahora() - timedelta(days=40))
    sub = _sub(nbt=_ahora() + timedelta(days=30), pagado_en=_ahora(), plan_overridden=False, custom_id=_USER)
    entorno.paypal.subs[_SUB] = sub

    res = _webhook(_ACTIVATED, sub, "TX-714-O3")

    assert res == {"success": True}
    fila = entorno.db.profiles[_USER]
    assert fila["paypal_subscription_id"] == _SUB
    assert fila["plan_tier"] == "plus" and fila["subscription_status"] == "ACTIVE"
    assert fila["subscription_end_date"] is None, "la fecha de fin de la sub vieja no es de esta"
    recuperada = entorno.alertas_de(f"billing_orphan_subscription_recovered:{_USER}:{_SUB}")
    assert recuperada and recuperada[0]["metadata"]["sub_reemplazada"] == "I-VIEJA"
    assert not entorno.alertas_de(f"billing_orphan_subscription_unrecoverable:{_SUB}")


@pytest.mark.parametrize("estado_vivo", ["ACTIVE", "PAYMENT_RETRYING", "APPROVED"])
def test_una_sub_vieja_viva_sigue_sin_pisarse(entorno, estado_vivo):
    """Guard de regresión (P1-BILLING-UPGRADE-FAIL-LOUD): pisar una sub que aún cobra es el doble cobro."""
    entorno.db.add_profile(plan_tier="basic", subscription_status=estado_vivo, paypal_subscription_id="I-VIEJA")
    sub = _sub(nbt=_ahora() + timedelta(days=30), pagado_en=_ahora(), custom_id=_USER)
    entorno.paypal.subs[_SUB] = sub

    _webhook(_ACTIVATED, sub, "TX-714-O4")

    fila = entorno.db.profiles[_USER]
    assert fila["paypal_subscription_id"] == "I-VIEJA" and fila["plan_tier"] == "basic"
    assert entorno.alertas_de(f"billing_orphan_subscription_unrecoverable:{_SUB}")


# ===========================================================================
# 4. `subscription_end_date` se limpia cuando una suscripción vuelve a estar activa
# ===========================================================================
def test_verify_limpia_la_fecha_de_fin_de_la_suscripcion_anterior(entorno, cliente):
    entorno.db.add_profile(plan_tier="plus", subscription_status="CANCELLED", paypal_subscription_id="I-VIEJA",
                           subscription_end_date=_ahora() + timedelta(days=5))
    entorno.paypal.subs["I-NUEVA"] = _sub("I-NUEVA", nbt=_ahora() + timedelta(days=30), pagado_en=_ahora())

    r = cliente.post("/api/subscription/verify",
                     json={"user_id": _USER, "subscriptionID": "I-NUEVA", "tier": "plus"})

    assert r.status_code == 200, r.text
    fila = entorno.db.profiles[_USER]
    assert fila["paypal_subscription_id"] == "I-NUEVA" and fila["subscription_status"] == "ACTIVE"
    assert fila["subscription_end_date"] is None, "Configuración pintaba esa fecha como «Próximo cobro»"


def test_reactivacion_por_webhook_limpia_la_fecha_de_fin(entorno):
    entorno.db.add_profile(plan_tier="gratis", subscription_status="INACTIVE", paypal_subscription_id=_SUB,
                           subscription_end_date=_ahora() - timedelta(days=10))

    _webhook(_ACTIVATED, _sub(nbt=_ahora() + timedelta(days=30), pagado_en=_ahora()), "TX-714-R1")

    fila = entorno.db.profiles[_USER]
    assert fila["subscription_status"] == "ACTIVE" and fila["plan_tier"] == "plus"
    assert fila["subscription_end_date"] is None


# ===========================================================================
# 5. GET /api/subscription/status
# ===========================================================================
_CLAVES_STATUS = {
    "has_paypal_subscription", "subscription_status", "paypal_status", "next_billing_time", "access_until",
    "plan_tier",
}


def test_status_sin_suscripcion(entorno, cliente):
    entorno.db.add_profile(plan_tier="gratis")

    r = cliente.get("/api/subscription/status")

    assert r.status_code == 200, r.text
    assert r.json() == {
        "has_paypal_subscription": False, "subscription_status": None, "paypal_status": None,
        "next_billing_time": None, "access_until": None, "plan_tier": "gratis",
    }
    assert entorno.paypal.calls == [], "sin suscripción no hay nada que preguntarle a PayPal"


def test_status_con_paypal_ok_y_cache(entorno, cliente):
    nbt = _ahora() + timedelta(days=12)
    _perfil_de_pago(entorno)
    entorno.paypal.subs[_SUB] = _sub(
        nbt=nbt, pagado_en=_ahora() - timedelta(days=18),
        subscriber={"email_address": "comprador-714@example.com", "name": {"given_name": "Ana"}},
    )

    r = cliente.get("/api/subscription/status")

    assert r.status_code == 200, r.text
    assert r.json() == {
        "has_paypal_subscription": True, "subscription_status": "ACTIVE", "paypal_status": "ACTIVE",
        "next_billing_time": nbt.isoformat(), "access_until": None, "plan_tier": "plus",
    }
    assert "comprador-714" not in r.text and "tok-714" not in r.text and "secreto-714" not in r.text
    llamadas = len(entorno.paypal.calls)
    r2 = cliente.get("/api/subscription/status")
    assert r2.json() == r.json()
    assert len(entorno.paypal.calls) == llamadas, "la segunda lectura sale de la caché, sin volver a PayPal"


def test_status_con_paypal_caido_es_fail_soft(entorno, cliente):
    _perfil_de_pago(entorno)
    entorno.paypal.fail["oauth"] = httpx.ConnectTimeout("connect timed out")

    r = cliente.get("/api/subscription/status")

    assert r.status_code == 200, r.text
    assert r.json() == {
        "has_paypal_subscription": True, "subscription_status": "ACTIVE", "paypal_status": None,
        "next_billing_time": None, "access_until": None, "plan_tier": "plus",
    }


def test_status_cancelada_da_access_until_y_ningun_proximo_cobro(entorno, cliente):
    fin = _ahora() + timedelta(days=9)
    _perfil_de_pago(entorno, subscription_status="CANCELLED", subscription_end_date=fin)
    entorno.paypal.subs[_SUB] = _sub(nbt=_ahora() + timedelta(days=9))  # PayPal aún no se enteró

    cuerpo = cliente.get("/api/subscription/status").json()

    assert set(cuerpo) == _CLAVES_STATUS
    assert cuerpo["access_until"] == fin.isoformat()
    assert cuerpo["next_billing_time"] is None, "una sub cancelada no tiene próximo cobro"
    assert cuerpo["subscription_status"] == "CANCELLED"


def test_status_no_espera_a_un_paypal_colgado(entorno, cliente, monkeypatch):
    """El timeout de httpx acota cada fase, no el total: el tope real es un `wait_for` sobre toda la consulta."""
    import time as _time
    monkeypatch.setattr(billing, "_STATUS_PAYPAL_TIMEOUT_S", 0.2)
    _perfil_de_pago(entorno)
    entorno.paypal.subs[_SUB] = _sub(nbt=_ahora() + timedelta(days=12))
    entorno.paypal.retraso_s = 5.0

    t0 = _time.monotonic()
    cuerpo = cliente.get("/api/subscription/status").json()

    assert _time.monotonic() - t0 < 2.0, "la tarjeta no puede quedarse esperando a PayPal"
    assert cuerpo["paypal_status"] is None and cuerpo["plan_tier"] == "plus"


def test_cancel_invalida_lo_que_status_tenia_cacheado(entorno, cliente):
    """Sin invalidar, la tarjeta seguiría viendo «ACTIVE» en PayPal hasta 10 minutos después de cancelar."""
    _perfil_de_pago(entorno)
    entorno.paypal.subs[_SUB] = _sub(nbt=_ahora() + timedelta(days=12))
    assert cliente.get("/api/subscription/status").json()["paypal_status"] == "ACTIVE"

    assert cliente.post("/api/subscription/cancel").status_code == 200
    cuerpo = cliente.get("/api/subscription/status").json()

    assert cuerpo["paypal_status"] == "CANCELLED"
    assert cuerpo["subscription_status"] == "CANCELLED" and cuerpo["next_billing_time"] is None
    assert cuerpo["access_until"] is not None


def test_status_sin_sesion_es_401(entorno, cliente):
    entorno.uid_jwt = None
    assert cliente.get("/api/subscription/status").status_code == 401


def test_status_solo_lee_la_fila_del_usuario_del_jwt(entorno, cliente):
    entorno.db.add_profile(plan_tier="gratis")
    entorno.db.add_profile(_OTHER, plan_tier="ultra", subscription_status="ACTIVE", paypal_subscription_id="I-AJENA")

    cuerpo = cliente.get("/api/subscription/status").json()

    assert cuerpo["plan_tier"] == "gratis" and cuerpo["has_paypal_subscription"] is False
    assert entorno.paypal.calls == []


def _dependencias(route):
    vistas = []

    def recorrer(dependant):
        for dep in dependant.dependencies:
            vistas.append(dep.call)
            recorrer(dep)

    recorrer(route.dependant)
    return vistas


@pytest.mark.parametrize(
    "ruta,metodo,tope",
    [("/api/subscription/status", "GET", 30), ("/api/subscription/cancel", "POST", 5)],
)
def test_jwt_y_limitador_sin_paywall(ruta, metodo, tope):
    """Cero IA ⇒ `get_verified_user_id` + RateLimiter, NUNCA `verify_api_quota` (un 402 dejaría al usuario sin poder
    ver ni cancelar lo que paga)."""
    route = next(r for r in billing.router.routes if r.path == ruta and metodo in r.methods)
    deps = _dependencias(route)
    assert get_verified_user_id in deps
    assert verify_api_quota not in deps
    limitadores = [d for d in deps if isinstance(d, RateLimiter)]
    assert limitadores and limitadores[0].max_calls == tope and limitadores[0].period == 60


# ===========================================================================
# Unidades: la aritmética del período pagado
# ===========================================================================
def test_sumar_meses_recorta_al_ultimo_dia_del_mes():
    utc = timezone.utc
    assert billing._add_months(datetime(2027, 1, 31, 9, tzinfo=utc), 1) == datetime(2027, 2, 28, 9, tzinfo=utc)
    assert billing._add_months(datetime(2028, 1, 31, 9, tzinfo=utc), 1) == datetime(2028, 2, 29, 9, tzinfo=utc)
    assert billing._add_months(datetime(2028, 2, 29, 9, tzinfo=utc), 12) == datetime(2029, 2, 28, 9, tzinfo=utc)
    assert billing._add_months(datetime(2026, 12, 15, tzinfo=utc), 1) == datetime(2027, 1, 15, tzinfo=utc)


def test_intervalo_del_plan_y_ante_la_duda_el_corto(monkeypatch):
    for clave in [k for k in os.environ if k.startswith("PAYPAL_PLAN_")]:
        monkeypatch.delenv(clave, raising=False)
    monkeypatch.setenv("PAYPAL_PLAN_BASIC_ID", "P-B")
    monkeypatch.setenv("PAYPAL_PLAN_ULTRA_ANNUAL_ID", "P-U-Y")
    assert billing._plan_interval_months("P-B") == 1
    assert billing._plan_interval_months("P-U-Y") == 12
    assert billing._plan_interval_months("P-DESCONOCIDO") is None
    assert billing._plan_interval_months(None) is None
    monkeypatch.setenv("PAYPAL_PLAN_PLUS_ID", "P-DOBLE")
    monkeypatch.setenv("PAYPAL_PLAN_PLUS_ANNUAL_ID", "P-DOBLE")
    assert billing._plan_interval_months("P-DOBLE") == 1, "nunca conceder más de lo pagado"


def test_next_billing_time_manda_sobre_el_calculo():
    utc = timezone.utc
    sub = {"plan_id": "P-X", "billing_info": {"next_billing_time": "2026-11-02T10:00:00Z",
                                              "last_payment": {"time": "2026-10-02T12:00:00Z"}}}
    assert billing._paid_through(sub) == (datetime(2026, 11, 2, 10, tzinfo=utc), "next_billing_time")
    assert billing._paid_through({"billing_info": {}}) == (None, "indeterminada")
    assert billing._paid_through(None) == (None, "indeterminada")


@pytest.mark.parametrize(
    "etapa_fallida,excepcion",
    [("cancel", httpx.ReadTimeout("read timed out")), ("oauth", httpx.ConnectError("connection refused"))],
    ids=["timeout-en-el-cancel", "red-caida-en-oauth"],
)
def test_borrar_cuenta_con_paypal_sin_respuesta_alerta_y_502(entorno, etapa_fallida, excepcion):
    """[P1-PLAN-LOTE-714] `cancel_paypal_subscription_for_user` (borrado de cuenta) tenía el mismo hueco que `/cancel`:
    una excepción de httpx salía como 500 genérico y sin alerta. Ahora: alerta crítica + 502, y el borrado se aborta."""
    entorno.paypal.subs[_SUB] = _sub(nbt=_ahora() + timedelta(days=15))
    entorno.paypal.fail[etapa_fallida] = excepcion

    with pytest.raises(billing.HTTPException) as exc:
        asyncio.run(billing.cancel_paypal_subscription_for_user(_USER, _SUB, reason="Cuenta eliminada"))

    assert exc.value.status_code == 502
    alertas = entorno.alertas_de(f"billing_cancel_failed:{_USER}:{_SUB}")
    assert len(alertas) == 1, f"alertas vistas: {entorno.claves()}"
    assert alertas[0]["severity"] == "critical"
    assert alertas[0]["metadata"]["flow"] == "account_delete"
    assert alertas[0]["metadata"]["stage"] == "network"
