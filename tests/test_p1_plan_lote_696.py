# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-696 · 2026-09-28] El coach no repite la pregunta de la comida que falta.

Auditoría del chat: «¿Ese almuerzo lo comiste o de verdad te lo saltaste?» salió dos turnos seguidos (22-sep, 02:23 y
02:29) sin que el dueño contestara la primera: la nota del lote 76 acompaña CADA registro o corrección y nada decía que
ya se había preguntado."""
from __future__ import annotations

import pytest

import tools

_TIPICAS = {"desayuno": 9.0, "almuerzo": 13.0, "merienda": 16.0, "cena": 19.5}


@pytest.fixture
def dia(monkeypatch):
    import db_facts
    monkeypatch.setattr(tools, "_local_date_str_for_user", lambda _u: "2026-09-22")
    monkeypatch.setattr(tools, "user_tz_offset_min", lambda _u: 240)
    monkeypatch.setattr(tools, "get_latest_usable_meal_plan", lambda _u: None)
    monkeypatch.setattr(tools, "_horas_tipicas_de_comida", lambda _u: dict(_TIPICAS))
    monkeypatch.setattr(db_facts, "get_consumed_meals_today",
                        lambda _u, date_str=None, tz_offset_mins=None: [{"meal_type": "desayuno"}, {"meal_type": "cena"}])


def test_hoy_con_la_comida_pasada_pide_no_insistir(dia):
    nota = tools._nota_comidas_sin_registrar("u-1", 0, ahora_local=22.5)
    assert "sigue sin registrar" in nota and "almuerzo" in nota
    assert "Una sola vez: si ya se lo preguntaste en esta conversación y no lo contestó, no lo repitas." in nota


def test_otro_dia_tambien(dia):
    nota = tools._nota_comidas_sin_registrar("u-1", 1, ahora_local=22.5)
    assert "ayer sigue sin registrar" in nota
    assert tools._NO_INSISTAS.strip() in nota


def test_sin_nada_pendiente_no_hay_nada_que_repetir(dia, monkeypatch):
    import db_facts
    monkeypatch.setattr(db_facts, "get_consumed_meals_today", lambda _u, date_str=None, tz_offset_mins=None: [
        {"meal_type": m} for m in ("desayuno", "almuerzo", "merienda", "cena")])
    nota = tools._nota_comidas_sin_registrar("u-1", 1, ahora_local=22.5)
    assert "ya tiene todas sus comidas registradas" in nota and "Una sola vez" not in nota


def test_marker():
    import app
    assert "P1-PLAN-LOTE-69" in app._LAST_KNOWN_PFIX
