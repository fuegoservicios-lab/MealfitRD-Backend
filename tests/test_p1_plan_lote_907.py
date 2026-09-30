"""[P1-PLAN-LOTE-907 · 2026-09-30] Anotar agua con la tarjeta de Hidratación apagada la enciende.

El dueño, probando GPT-Live-1: «me bebí 3 vasos de agua» → «sumé tus 3 vasos, llevas 4 de 9», y la Hidratación seguía
apagada: los vasos quedaban guardados donde no se ven. Sin red: la base y los recordatorios son falsos.
"""
from __future__ import annotations

import pytest

import ajustes_de_la_app as aj


@pytest.fixture
def perfil(monkeypatch):
    import db_profiles
    import hydration_reminders
    estado = {"encendida": False, "escrituras": [], "recordatorios": 0}
    monkeypatch.setattr(db_profiles, "get_water_tracker_enabled", lambda uid: estado["encendida"])

    def _poner(uid, valor):
        estado["escrituras"].append((uid, valor))
        estado["encendida"] = valor
        return True

    monkeypatch.setattr(db_profiles, "update_water_tracker_enabled", _poner)
    monkeypatch.setattr(hydration_reminders, "al_encender",
                        lambda uid: estado.__setitem__("recordatorios", estado["recordatorios"] + 1))
    return estado


def test_apagada_se_enciende_y_la_pantalla_se_entera(perfil):
    extra = aj.encender_hidratacion_al_anotar_agua("u1")
    texto, cambio = aj.extraer_marcador("Listo: se sumaron 3 vaso(s)." + extra)
    assert perfil["escrituras"] == [("u1", True)] and perfil["recordatorios"] == 1
    assert cambio == {"hidratacion": True}, "el marcador llega al `done` y a las novedades de GPT-Live-1"
    assert "encendí" in texto and "<<" not in texto, "el modelo lo dice, sin ver el JSON"


def test_encendida_o_invitado_no_toca_nada(perfil):
    perfil["encendida"] = True
    assert aj.encender_hidratacion_al_anotar_agua("u1") == ""
    perfil["encendida"] = False
    assert aj.encender_hidratacion_al_anotar_agua("guest") == ""
    assert aj.encender_hidratacion_al_anotar_agua(None) == ""
    assert perfil["escrituras"] == []


def test_si_la_base_falla_el_vaso_sigue_anotado(monkeypatch):
    import db_profiles
    monkeypatch.setattr(db_profiles, "get_water_tracker_enabled", lambda uid: (_ for _ in ()).throw(RuntimeError("x")))
    assert aj.encender_hidratacion_al_anotar_agua("u1") == ""


def test_log_water_glass_la_enciende_al_sumar_y_no_al_restar(perfil, monkeypatch):
    import db
    import tools
    monkeypatch.setattr(db, "connection_pool", object())
    monkeypatch.setattr(db, "execute_sql_write", lambda *a, **k: [{"glasses": 4}])
    monkeypatch.setattr(tools, "_local_date_str_for_user", lambda uid: "2026-09-30")
    restado = tools.log_water_glass.invoke({"user_id": "u1", "count_delta": -1})
    assert aj.extraer_marcador(restado)[1] is None and perfil["escrituras"] == []
    sumado = tools.log_water_glass.invoke({"user_id": "u1", "count_delta": 3})
    assert aj.extraer_marcador(sumado)[1] == {"hidratacion": True}
    assert perfil["escrituras"] == [("u1", True)]
