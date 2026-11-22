# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-1166 · 2026-11-22] Eliminar cuentas en lote desde el panel: dos pasos y frenos que no dependen de la
buena vista del admin (tope, «ELIMINAR <n>», nunca admins ni suscripciones vivas, activas solo con su casilla)."""
import pytest

import admin_eliminar_cuentas as aec

A, B, C, D = ("11111111-1111-1111-1111-111111111111", "22222222-2222-2222-2222-222222222222",
              "33333333-3333-3333-3333-333333333333", "44444444-4444-4444-4444-444444444444")


@pytest.fixture
def mundo(monkeypatch):
    filas = {
        A: {"email": "vacia@x.com", "activa": False, "comidas_total": 0},
        B: {"email": "activa@x.com", "activa": True, "comidas_total": 120},
        C: {"email": "paga@x.com", "activa": False, "paypal_subscription_id": "I-1", "subscription_status": "ACTIVE"},
        D: {"email": "admin@x.com", "activa": False},
    }
    rastro, borradas = [], []
    monkeypatch.setattr(aec, "_filas", lambda ids: {i: filas[i] for i in ids if i in filas})
    monkeypatch.setattr(aec, "es_admin", lambda uid: uid == D)
    monkeypatch.setattr(aec, "registrar_acceso", lambda a, acc, uid, det: rastro.append((acc, uid)))
    monkeypatch.setattr(aec, "_borrar_una", lambda uid: borradas.append(uid) or {"failed_steps": []})
    return rastro, borradas


def test_revisar_dice_bloqueos_y_actividad_sin_borrar(mundo):
    rastro, borradas = mundo
    r = {x["user_id"]: x for x in aec.revisar([A, B, C, D, "no-uuid", "55555555-5555-5555-5555-555555555555"])}
    assert r[A]["bloqueo"] is None and r[A]["activa"] is False
    assert r[B]["activa"] is True and r[C]["bloqueo"] == "suscripcion_activa" and r[D]["bloqueo"] == "es_admin"
    assert r["55555555-5555-5555-5555-555555555555"]["bloqueo"] == "no_existe"
    assert rastro == [] and borradas == []


def test_confirmacion_escrita_con_el_numero(mundo):
    for mala in ("ELIMINAR", "eliminar 3", "ELIMINAR 1", ""):
        with pytest.raises(aec.ErrorEliminar) as e:
            aec.eliminar("adm", [A, B], mala)
        assert e.value.detalle == "confirmacion"
    assert mundo[1] == []


def test_tope_de_lote():
    ids = [f"00000000-0000-0000-0000-{i:012d}" for i in range(aec.MAX_LOTE + 1)]
    with pytest.raises(aec.ErrorEliminar) as e:
        aec.eliminar("adm", ids, f"ELIMINAR {len(ids)}")
    assert e.value.detalle == "demasiadas"


def test_activa_solo_con_su_casilla_y_nunca_admin_ni_pago(mundo):
    rastro, borradas = mundo
    res = {x["user_id"]: x["resultado"] for x in aec.eliminar("adm", [A, B, C, D], "eliminar 4")}
    assert res == {A: "eliminada", B: "activa_sin_confirmar", C: "suscripcion_activa", D: "es_admin"}
    assert borradas == [A] and rastro == [("eliminar_cuenta", A)]
    res2 = {x["user_id"]: x["resultado"] for x in aec.eliminar("adm", [B], "ELIMINAR 1", confirmar_activas=[B])}
    assert res2 == {B: "eliminada"} and borradas == [A, B]


def test_sin_rastro_no_se_borra(mundo, monkeypatch):
    _, borradas = mundo

    def falla(*a):
        raise RuntimeError("db")
    monkeypatch.setattr(aec, "registrar_acceso", falla)
    with pytest.raises(aec.ErrorEliminar) as e:
        aec.eliminar("adm", [A], "ELIMINAR 1")
    assert e.value.status == 503 and borradas == []


def test_rutas_del_panel_registradas():
    from routers.admin import router
    rutas = {r.path for r in router.routes}
    assert {"/api/admin/cuentas/eliminar/revisar", "/api/admin/cuentas/eliminar", "/api/admin/pruebas/lote/quitar"} <= rutas \
        or {"/cuentas/eliminar/revisar", "/cuentas/eliminar", "/pruebas/lote/quitar"} <= {p.split("/api/admin")[-1] for p in rutas}
