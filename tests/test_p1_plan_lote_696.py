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
    monkeypatch.setattr(tools, "_diary_date_str_for_user", lambda _u=None: "2026-09-22")  # [P1-PLAN-LOTE-980]
    monkeypatch.setattr(tools, "user_tz_offset_min", lambda _u: 240)
    monkeypatch.setattr(tools, "get_latest_usable_meal_plan", lambda _u: None)
    monkeypatch.setattr(tools, "_horas_tipicas_de_comida", lambda _u: dict(_TIPICAS))
    monkeypatch.setattr(db_facts, "get_consumed_meals_today",
                        lambda _u, date_str=None, tz_offset_mins=None: [{"meal_type": "desayuno", "calories": 400}, {"meal_type": "cena", "calories": 400}])


def test_hoy_con_la_comida_pasada_pide_no_insistir(dia):
    # [P1-PLAN-LOTE-980] BB-048/BB-051: pasadas las 21:00 con la cena ya registrada, el día cerró y
    # no se insiste en el almuerzo que faltó. A media tarde, la comida pasada sí se ofrece una vez.
    nota = tools._nota_comidas_sin_registrar("u-1", 0, ahora_local=22.5)
    assert "el día ya cerró" in nota and "sigue sin registrar" not in nota
    nota = tools._nota_comidas_sin_registrar("u-1", 0, ahora_local=17.0)
    assert "sigue sin registrar" in nota and "almuerzo" in nota
    assert "Una sola vez: si ya se lo preguntaste en esta conversación y no lo contestó, no lo repitas." in nota


def test_otro_dia_tambien(dia):
    nota = tools._nota_comidas_sin_registrar("u-1", 1, ahora_local=22.5)
    assert "Comida de ayer registrada correctamente. Confirma la anotación de forma limpia sin preguntar por otras comidas faltantes de ayer." in nota  # BB-044 [P1-PLAN-LOTE-980]


def test_sin_nada_pendiente_no_hay_nada_que_repetir(dia, monkeypatch):
    import db_facts
    monkeypatch.setattr(db_facts, "get_consumed_meals_today", lambda _u, date_str=None, tz_offset_mins=None: [
        {"meal_type": m, "calories": 400} for m in ("desayuno", "almuerzo", "merienda", "cena")])
    nota = tools._nota_comidas_sin_registrar("u-1", 0, ahora_local=15.0)  # [P1-PLAN-LOTE-980] hoy: ayer es BB-044
    assert "ya tiene todas sus comidas registradas" in nota and "Una sola vez" not in nota


def test_marker():
    import app
    # [P1-PLAN-LOTE-713] «el marker no bajó de 696», no «el marker empieza por 69»: el siguiente
    # bloque de otra sesión lo sube a 7xx y el prefijo fallaba sin que nada se hubiera roto.
    import re
    m = re.match(r"P1-PLAN-LOTE-(\d+) ", app._LAST_KNOWN_PFIX)
    assert m and int(m.group(1)) >= 696, app._LAST_KNOWN_PFIX
