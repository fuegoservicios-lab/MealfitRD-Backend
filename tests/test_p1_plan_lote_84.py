# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-84 · 2026-09-17] El cierre tras registrar sabe qué hora es. El dueño registró el desayuno a las 11:23 y el
coach cerró con «¿Te anoto el almuerzo cuando lo comas?»: la nota del lote 76 ofrecía la comida que falta sin mirar si ya
tocaba. Ahora lo que aún no ha llegado se cierra con una frase consciente de la hora y sin pregunta; lo que ya pasó su
hora se ofrece por su nombre como antes; ayer no cambia."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

import tools

_BACKEND = Path(__file__).resolve().parents[1]
_TIPICAS = {"desayuno": 9.0, "almuerzo": 13.0, "merienda": 16.0, "cena": 19.5}


def test_separa_lo_que_ya_paso_de_lo_que_aun_no_toca():
    assert tools._comidas_por_hora(["almuerzo", "merienda", "cena"], 11.38, _TIPICAS) == ([], ["almuerzo", "merienda", "cena"])
    assert tools._comidas_por_hora(["almuerzo", "merienda", "cena"], 14.0, _TIPICAS) == (["almuerzo"], ["merienda", "cena"])
    assert tools._comidas_por_hora(["desayuno", "cena"], 21.0, _TIPICAS) == (["desayuno", "cena"], [])
    assert tools._comidas_por_hora(["almuerzo"], 13.5, _TIPICAS) == ([], ["almuerzo"])   # 45 min de gracia tras su hora


def test_formato_de_la_hora():
    assert tools._fmt_hora_float(11.38) == "11:23 AM"
    assert tools._fmt_hora_float(13.0) == "1:00 PM"
    assert tools._fmt_hora_float(0.25) == "12:15 AM"
    assert tools._fmt_hora_float(19.5) == "7:30 PM"


@pytest.fixture
def dia_del_dueno(monkeypatch):
    import db_facts
    monkeypatch.setattr(tools, "_local_date_str_for_user", lambda _u: "2026-09-17")
    monkeypatch.setattr(tools, "user_tz_offset_min", lambda _u: 240)
    monkeypatch.setattr(tools, "get_latest_usable_meal_plan", lambda _u: None)
    monkeypatch.setattr(tools, "_horas_tipicas_de_comida", lambda _u: dict(_TIPICAS))
    monkeypatch.setattr(db_facts, "get_consumed_meals_today",
                        lambda _u, date_str=None, tz_offset_mins=None: [{"meal_type": "desayuno", "meal_name": "huevos"}])


def test_a_las_11_23_el_almuerzo_aun_no_toca_y_no_se_pregunta(dia_del_dueno):
    nota = tools._nota_comidas_sin_registrar("u-1", 0, ahora_local=11.38)
    assert "son las 11:23 AM" in nota
    assert "La próxima es almuerzo (~1:00 PM)" in nota
    assert "NO preguntes si la anotas" in nota
    assert "ya casi es hora de almorzar: cuando almuerces, cuéntame qué comiste y lo anoto" in nota
    assert "ofreciendo agregar LA QUE FALTA" not in nota


def test_a_las_9_la_cena_queda_lejos_y_la_frase_no_dice_ya_casi(dia_del_dueno):
    import db_facts
    db_facts.get_consumed_meals_today = lambda _u, date_str=None, tz_offset_mins=None: [
        {"meal_type": "desayuno"}, {"meal_type": "almuerzo"}]
    nota = tools._nota_comidas_sin_registrar("u-1", 0, ahora_local=14.5)
    assert "La próxima es cena (~7:30 PM)" in nota
    assert "«cuando cenes, cuéntame qué comiste y lo anoto»" in nota and "ya casi" not in nota


def test_a_las_2_de_la_tarde_el_almuerzo_ya_paso_y_se_ofrece_por_su_nombre(dia_del_dueno):
    nota = tools._nota_comidas_sin_registrar("u-1", 0, ahora_local=14.0)
    assert "son las 2:00 PM" in nota
    assert "sigue sin registrar almuerzo, que ya pasó su hora: cierra ofreciendo agregarla por su nombre" in nota
    assert "cena aún no toca: no preguntes por ella" in nota


def test_ayer_no_cambia(dia_del_dueno):
    nota = tools._nota_comidas_sin_registrar("u-1", 1, ahora_local=11.38)
    assert "ayer sigue sin registrar" in nota and "ofreciendo agregar LA QUE FALTA" in nota


def test_todo_registrado_no_cambia(dia_del_dueno):
    import db_facts
    db_facts.get_consumed_meals_today = lambda _u, date_str=None, tz_offset_mins=None: [
        {"meal_type": "desayuno"}, {"meal_type": "almuerzo"}, {"meal_type": "cena"}]
    nota = tools._nota_comidas_sin_registrar("u-1", 0, ahora_local=21.0)
    assert "ya tiene todas sus comidas registradas" in nota


def test_sin_hora_local_cae_a_la_nota_de_antes(dia_del_dueno, monkeypatch):
    monkeypatch.setattr(tools, "_hora_local_float", lambda _u: None)
    nota = tools._nota_comidas_sin_registrar("u-1", 0)
    assert "ofreciendo agregar LA QUE FALTA" in nota


def test_marcador_y_documento():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · 2026-', (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 84
    assert "P1-PLAN-LOTE-84" in (_BACKEND / "docs" / "coach_bateria_2026_09_15.md").read_text(encoding="utf-8")
