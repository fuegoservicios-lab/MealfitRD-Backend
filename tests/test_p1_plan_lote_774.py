"""[P1-PLAN-LOTE-774 · 2026-09-28] Panel · Cuentas: buscar por correo exacto, ficha sin datos de salud, regalar
créditos o una cortesía y revertir. El rastro va ANTES de escribir; si no se anota, no hay cambio."""
import json
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import admin_cuentas as ac
import auth
import regalos_cuenta as rc
import routers.admin as ra
from auth import get_verified_user_id

_BACKEND = Path(__file__).resolve().parents[1]
ADMIN = "11111111-1111-1111-1111-111111111111"
UID = "33333333-3333-3333-3333-333333333333"
GID = "44444444-4444-4444-4444-444444444444"
RD = ZoneInfo("America/Santo_Domingo")
H = {"X-Admin-Accion": "1"}


class _BD:
    def __init__(self):
        self.plan, self.usados, self.usados_coach = "gratis", 3, 0
        self.vigentes, self.historial = [], []
        self.orden, self.escrituras, self.transacciones, self.rastro, self.avisos = [], [], [], [], []

    def query(self, q, p=None, fetch_one=False, fetch_all=False):
        q = " ".join(q.split())
        if "FROM public.user_profiles WHERE lower(email)" in q:
            return {"id": UID} if p[0] == "ana@correo.com" else None
        if "FROM public.user_profiles WHERE id" in q:
            return {"id": UID, "email": "ana@correo.com", "full_name": "Ana",
                    "created_at": datetime(2026, 9, 1, tzinfo=timezone.utc), "plan_tier": self.plan,
                    "subscription_status": None, "subscription_end_date": None, "tiene_paypal": False}
        if "FROM public.account_grants WHERE user_id" in q:
            return self.historial
        if "FROM public.account_grants WHERE id" in q:
            return next((h for h in self.historial if h["id"] == p[0]), None)
        raise AssertionError(q[:80])


@pytest.fixture
def bd(monkeypatch):
    b = _BD()

    def _escribir(q, p=None, **k):
        b.orden.append("escritura")
        b.escrituras.append((" ".join(q.split()), p))
        return [{"id": p[2]}] if k.get("returning") else True

    def _anotar(admin, accion, objetivo=None, detalle=None):
        b.orden.append("rastro")
        b.rastro.append((accion, objetivo, detalle))

    monkeypatch.setattr(ac, "execute_sql_query", b.query)
    monkeypatch.setattr(ac, "execute_sql_write", _escribir)
    monkeypatch.setattr(ac, "execute_sql_transaction", lambda pares: b.transacciones.append(pares) or True)
    monkeypatch.setattr(ac, "get_monthly_api_usage",
                        lambda uid, kind="generation": b.usados_coach if kind == "coach" else b.usados)
    monkeypatch.setattr(ac, "registrar_acceso", _anotar)
    monkeypatch.setattr(ac, "avisar_en_segundo_plano", lambda uid, r: b.avisos.append((uid, r)))
    monkeypatch.setattr(ac, "_invalidar_plan", lambda uid: b.avisos.append(("cache", uid)))
    monkeypatch.setattr(rc, "regalos_vigentes", lambda uid: b.vigentes)
    return b


def _vigente(kind, amount=None, plan=None):
    return {"id": "g", "kind": kind, "amount": amount, "plan": plan, "starts_at": None,
            "ends_at": rc.inicio_de_mes(1) if kind != "plan" else None, "created_at": datetime.now(timezone.utc)}


def test_buscar_normaliza_el_correo(bd):
    assert ac.buscar_por_correo("  Ana@Correo.COM ") == UID
    assert ac.buscar_por_correo("nadie@correo.com") is None
    assert ac.buscar_por_correo("sin-arroba") is None


def test_la_ficha_suma_regalos_y_no_trae_salud(bd):
    bd.vigentes = [_vigente("creditos_generacion", 20)]
    f = ac.ficha(UID)
    lim = auth._TIER_LIMITS["gratis"]
    assert f["creditos"] == {"usados": 3, "plan": lim, "regalo": 20, "tope": lim + 20}
    assert f["plan_efectivo"] == "gratis" and f["cortesia"] is None and f["es_admin"] is False
    assert "health" not in json.dumps(f, default=str)


def test_regalar_creditos_anota_antes_de_escribir_y_avisa(bd):
    r = ac.regalar_creditos(ADMIN, UID, "generacion", "sumar", 20, "mes", "compensación")
    assert bd.orden == ["rastro", "escritura"]
    accion, objetivo, detalle = bd.rastro[0]
    assert accion == "regalar_creditos" and objetivo == UID and detalle["cantidad"] == 20
    assert detalle["despues"]["tope"] == detalle["antes"]["tope"] + 20
    sql, p = bd.escrituras[0]
    assert "INSERT INTO public.account_grants" in sql and p[2] == "creditos_generacion" and p[3] == 20
    assert p[4] == rc.inicio_de_mes(1) and r["cantidad"] == 20
    assert bd.avisos and bd.avisos[0][0] == UID


def test_si_el_rastro_falla_no_hay_cambio(bd, monkeypatch):
    def _roto(*a, **k):
        raise RuntimeError("sin DB")
    monkeypatch.setattr(ac, "registrar_acceso", _roto)
    with pytest.raises(ac.ErrorRegalo) as ei:
        ac.regalar_creditos(ADMIN, UID, "generacion", "sumar", 5, "mes", "compensación")
    assert ei.value.status == 503 and bd.escrituras == [] and bd.avisos == []


def test_recargar_al_completo_deja_el_cupo_del_plan(bd):
    # Review Focus 2: gastó 15 con 5 regalados → se regalan 10 y quedan disponibles exactamente los del plan
    bd.usados = 15
    bd.vigentes = [_vigente("creditos_generacion", 5)]
    assert ac.regalar_creditos(ADMIN, UID, "generacion", "completo", None, "mes", "compensación")["cantidad"] == 10
    bd.usados = 4
    with pytest.raises(ac.ErrorRegalo) as ei:
        ac.regalar_creditos(ADMIN, UID, "generacion", "completo", None, "mes", "otra vez")
    assert ei.value.status == 409


@pytest.mark.parametrize("medidor,modo,cantidad,hasta,motivo", [
    ("generacion", "sumar", 0, "mes", "motivo ok"), ("generacion", "sumar", 1001, "mes", "motivo ok"),
    ("generacion", "sumar", "x", "mes", "motivo ok"), ("otro", "sumar", 5, "mes", "motivo ok"),
    ("generacion", "otro", 5, "mes", "motivo ok"), ("generacion", "sumar", 5, "siempre", "motivo ok"),
    ("generacion", "sumar", 5, "mes", "no"),
])
def test_validaciones_de_creditos(bd, medidor, modo, cantidad, hasta, motivo):
    with pytest.raises(ac.ErrorRegalo) as ei:
        ac.regalar_creditos(ADMIN, UID, medidor, modo, cantidad, hasta, motivo)
    assert ei.value.status == 422 and bd.escrituras == []


def test_a_una_cuenta_admin_no_se_le_regala(bd):
    bd.plan = "admin"
    with pytest.raises(ac.ErrorRegalo) as ei:
        ac.regalar_creditos(ADMIN, UID, "generacion", "sumar", 5, "mes", "intento")
    assert ei.value.status == 409


def test_cortesia_mejor_que_lo_pagado_y_hasta_el_dia_incluido(bd):
    bd.plan = "basic"
    hoy = datetime.now(RD).date()
    ac.dar_cortesia(ADMIN, UID, "plus", (hoy + timedelta(days=30)).isoformat(), "tester del beta")
    (upd, _), (ins, p_ins) = bd.transacciones[0]
    assert "UPDATE public.account_grants SET revoked_at = now()" in upd and "kind = 'plan' AND revoked_at IS NULL" in upd
    assert "INSERT INTO public.account_grants" in ins and p_ins[2] == "plus"
    assert p_ins[3] == datetime.combine(hoy + timedelta(days=31), time(0), tzinfo=RD)
    assert ("cache", UID) in bd.avisos and bd.rastro[0][0] == "dar_cortesia"


def test_cortesia_sin_fecha(bd):
    ac.dar_cortesia(ADMIN, UID, "ultra", None, "tester del beta")
    assert bd.transacciones[0][1][1][3] is None


def test_cortesia_igual_o_peor_que_lo_pagado_409(bd):
    bd.plan = "plus"
    with pytest.raises(ac.ErrorRegalo) as ei:
        ac.dar_cortesia(ADMIN, UID, "plus", None, "no aplica")
    assert ei.value.status == 409 and bd.transacciones == []


@pytest.mark.parametrize("dias", [-1, 367])
def test_fecha_de_cortesia_fuera_de_rango(bd, dias):
    hoy = datetime.now(RD).date()
    with pytest.raises(ac.ErrorRegalo) as ei:
        ac.dar_cortesia(ADMIN, UID, "plus", (hoy + timedelta(days=dias)).isoformat(), "fuera de rango")
    assert ei.value.status == 422


def test_admin_no_es_un_plan_regalable(bd):
    with pytest.raises(ac.ErrorRegalo) as ei:
        ac.dar_cortesia(ADMIN, UID, "admin", None, "intento")
    assert ei.value.status == 422


def test_dos_cortesias_a_la_vez_la_segunda_recibe_409(bd, monkeypatch):
    # Review Focus 1: el índice único corta a la segunda; nunca quedan dos vivas
    class UniqueViolation(Exception):
        pass

    def _choca(pares):
        raise UniqueViolation("duplicate key value violates unique constraint")
    monkeypatch.setattr(ac, "execute_sql_transaction", _choca)
    with pytest.raises(ac.ErrorRegalo) as ei:
        ac.dar_cortesia(ADMIN, UID, "plus", None, "carrera")
    assert ei.value.status == 409 and bd.avisos == []          # ni aviso ni caché: no se dio nada
    assert any(r[0] == "dar_cortesia_fallo" for r in bd.rastro)


def test_revertir(bd):
    bd.historial = [{"id": GID, "user_id": UID, "kind": "plan", "revoked_at": None}]
    r = ac.revocar(ADMIN, GID, "se acabó la prueba")
    assert r == {"user_id": UID, "grant_id": GID} and ("cache", UID) in bd.avisos
    assert "WHERE id = %s AND revoked_at IS NULL RETURNING id" in bd.escrituras[-1][0]
    assert bd.orden == ["rastro", "escritura"]


def test_revertir_lo_ya_revertido_o_inexistente(bd):
    bd.historial = [{"id": GID, "user_id": UID, "kind": "plan", "revoked_at": datetime.now(timezone.utc)}]
    with pytest.raises(ac.ErrorRegalo) as ei:
        ac.revocar(ADMIN, GID, "otra vez")
    assert ei.value.status == 409
    with pytest.raises(ac.ErrorRegalo) as ei:
        ac.revocar(ADMIN, "no-es-uuid", "motivo")
    assert ei.value.status == 404


def test_revertir_con_la_carrera_perdida_deja_su_fallo_en_el_rastro(bd, monkeypatch):
    # Revisión final: otro lo revirtió entre la lectura y el UPDATE → 409, y el rastro no se queda en «revocar_regalo»
    bd.historial = [{"id": GID, "user_id": UID, "kind": "plan", "revoked_at": None}]
    monkeypatch.setattr(ac, "execute_sql_write", lambda q, p=None, **k: [] if k.get("returning") else True)
    with pytest.raises(ac.ErrorRegalo) as ei:
        ac.revocar(ADMIN, GID, "se acabó la prueba")
    assert ei.value.status == 409 and ("cache", UID) not in bd.avisos
    assert [r[0] for r in bd.rastro] == ["revocar_regalo", "revocar_regalo_fallo"]
    assert bd.rastro[1][1] == UID and bd.rastro[1][2]["grant_id"] == GID

    def _fallo_sin_rastro(admin, accion, objetivo=None, detalle=None):      # best-effort: sin rastro sigue el 409
        if accion.endswith("_fallo"):
            raise RuntimeError("sin DB")
    monkeypatch.setattr(ac, "registrar_acceso", _fallo_sin_rastro)
    with pytest.raises(ac.ErrorRegalo) as ei:
        ac.revocar(ADMIN, GID, "se acabó la prueba")
    assert ei.value.status == 409


def test_revertir_funciona_con_el_knob_apagado(bd, monkeypatch):
    # Review Focus 4: apagado, regalar da 503 claro y revertir sigue funcionando
    monkeypatch.setenv("MEALFIT_ACCOUNT_GRANTS", "false")
    with pytest.raises(ac.ErrorRegalo) as ei:
        ac.regalar_creditos(ADMIN, UID, "generacion", "sumar", 5, "mes", "motivo")
    assert ei.value.status == 503
    bd.historial = [{"id": GID, "user_id": UID, "kind": "creditos_generacion", "revoked_at": None}]
    assert ac.revocar(ADMIN, GID, "motivo")["user_id"] == UID


@pytest.fixture
def cliente(monkeypatch, bd):
    monkeypatch.setenv("MEALFIT_ADMIN_PANEL", "true")
    monkeypatch.setenv("MEALFIT_ADMIN_USER_IDS", ADMIN)
    monkeypatch.setattr(ra, "registrar_acceso", lambda admin, accion, *a, **k: bd.rastro.append((accion,)))

    def _hacer(uid):
        app = FastAPI()
        app.include_router(ra.router)
        app.dependency_overrides[get_verified_user_id] = lambda: uid
        return TestClient(app)
    return _hacer


def test_quien_no_es_admin_recibe_404(cliente):
    otro = "22222222-2222-2222-2222-222222222222"
    assert cliente(otro).post("/api/admin/cuentas/buscar", json={"email": "ana@correo.com"}, headers=H).status_code == 404


def test_sin_la_cabecera_de_accion_403(cliente):
    assert cliente(ADMIN).post("/api/admin/cuentas/buscar", json={"email": "ana@correo.com"}).status_code == 403


def test_buscar_anota_y_devuelve_la_ficha(cliente, bd):
    r = cliente(ADMIN).post("/api/admin/cuentas/buscar", json={"email": "Ana@correo.com"}, headers=H)
    assert r.status_code == 200 and r.json()["cuenta"]["email"] == "ana@correo.com"
    assert ("buscar_cuenta",) in bd.rastro
    assert cliente(ADMIN).post("/api/admin/cuentas/buscar", json={"email": "x@y.z"}, headers=H).json() == {"cuenta": None}


def test_regalar_por_http_y_los_errores_con_su_codigo(cliente, bd):
    c = cliente(ADMIN)
    cuerpo = {"medidor": "generacion", "modo": "sumar", "cantidad": 20, "hasta": "mes", "motivo": "compensación"}
    r = c.post(f"/api/admin/cuentas/{UID}/creditos", json=cuerpo, headers=H)
    assert r.status_code == 200 and r.json()["ok"] is True and r.json()["cuenta"]["user_id"] == UID
    r = c.post(f"/api/admin/cuentas/{UID}/creditos", json={**cuerpo, "cantidad": 0}, headers=H)
    assert r.status_code == 422 and "1 a 1000" in r.json()["detail"]


def test_si_releer_la_ficha_falla_el_regalo_dado_no_es_un_500(cliente, monkeypatch):
    # Revisión final: el regalo YA se guardó; si releer la ficha falla → 200 con `cuenta: null` (el panel la vuelve a
    # pedir). Un 500 invitaría a reintentar y duplicar el regalo.
    monkeypatch.setattr(ac, "regalar_creditos", lambda *a: {"grant_id": GID, "cantidad": 20, "hasta": "2026-10-01"})

    def _ficha_rota(uid):
        raise RuntimeError("sin DB")
    monkeypatch.setattr(ac, "ficha", _ficha_rota)
    cuerpo = {"medidor": "generacion", "modo": "sumar", "cantidad": 20, "hasta": "mes", "motivo": "compensación"}
    r = cliente(ADMIN).post(f"/api/admin/cuentas/{UID}/creditos", json=cuerpo, headers=H)
    assert r.status_code == 200 and r.json() == {"ok": True, "grant_id": GID, "cantidad": 20, "hasta": "2026-10-01",
                                                 "cuenta": None}


def test_la_cabecera_de_accion_pasa_el_cors():
    # Revisión final: en desarrollo el panel es de OTRO origen (:5173 → :8000): sin la cabecera en `allow_headers` el
    # preflight corta todos los POST del panel.
    src = (_BACKEND / "app.py").read_text(encoding="utf-8")
    cors = src[src.index("    CORSMiddleware,"):]           # el comentario de arriba CITA `allow_headers=["*"]`
    i = cors.index("allow_headers=[")
    assert '"X-Admin-Accion"' in cors[i:cors.index("\n    ],", i)]


def test_las_rutas_no_llevan_el_segmento_reservado():
    assert all("/admin/" not in r.path.replace("/api/admin", "", 1) for r in ra.router.routes)


def test_las_metricas_cuentan_las_cortesias(monkeypatch):
    import admin_metricas as am
    from tests.test_p1_plan_lote_637 import _bloque, _con
    f, _ = _con(**{"FROM public.account_grants WHERE kind = 'plan'": {"n": 2}})
    monkeypatch.setattr(am, "admin_ids", lambda: frozenset())
    monkeypatch.setattr(am, "execute_sql_query", f)
    filas = {x["etiqueta"]: x["valor"] for x in _bloque(am.metricas(7), "cuentas")["filas"]}
    assert filas["Con plan de cortesía"] == "2"
