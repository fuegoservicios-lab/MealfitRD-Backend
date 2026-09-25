# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-226 · 2026-09-25] El coach decide con el margen que le queda al usuario, y el chat de hoy se recupera.

1. Caso del dueño (24-sep, 22:31): cena de claras «sin yemas» y la pregunta «¿me recomiendas comérmelas?». El coach:
   «no hay razón para sacarlas — dos yemas son ~110 kcal». Llevaba 52 de 57 g de grasa: las yemas le pasaban la meta.
   Las cifras las tenía, pero la grasa iba entre paréntesis, la resta la hacía el modelo y ninguna regla le pedía
   comprobar que lo recomendado CABE. Ahora `coach_day_context.margen_del_dia` da la resta macro a macro (en el bloque
   del día y en la nota tras registrar) con `REGLA_MARGEN` al lado, y la regla R2 del prompt lo exige.
2. «Volver al chat de hoy» (frontend: `src/__tests__/lote226.test.jsx`): abrir un chat viejo desde Recientes lo marcaba
   como el de hoy y «Nuevo chat» quedaba bloqueado: no había forma de volver.

Tooltip-anchor: P1-PLAN-LOTE-226
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]


def _src(rel: str) -> str:
    return (_BACKEND / rel).read_text(encoding="utf-8")


# El día del dueño antes de la cena: desayuno 415 kcal / G15 y almuerzo 715 kcal / G24, más la cena sin yemas (G13).
_METAS = {"kcal": 2050, "protein_g": 134, "carbs_g": 251, "fats_g": 57, "fuente": "contador"}
_DIARIO = [
    {"calories": 415, "protein": 24, "carbs": 48, "healthy_fats": 15},
    {"calories": 715, "protein": 35, "carbs": 91, "healthy_fats": 24},
    {"calories": 245, "protein": 19, "carbs": 16, "healthy_fats": 13},
]


def test_el_margen_resta_cada_macro_y_nombra_la_grasa_casi_agotada():
    import coach_day_context as cdc
    m = cdc.margen_del_dia(_METAS, cdc.consumido_hoy(_DIARIO))
    assert "kcal: quedan ~675" in m
    assert "proteína: quedan ~56 g" in m
    assert "grasas: quedan ~5 g" in m          # 57 − 52: las yemas (~10 g) no caben
    assert "Sin margen (o casi) en: grasas" in m
    assert "proteína" not in m.split("Sin margen")[1]   # la proteína que falta no es «sin margen»


def test_el_margen_dice_cuando_ya_se_paso():
    import coach_day_context as cdc
    m = cdc.margen_del_dia(_METAS, {"kcal": 2100, "protein_g": 140, "carbs_g": 200, "fats_g": 70, "registros": 4})
    assert "grasas: YA SE PASÓ por ~13 g" in m and "kcal: YA SE PASÓ por ~50" in m


def test_sin_metas_no_hay_margen():
    import coach_day_context as cdc
    assert cdc.margen_del_dia(None, cdc.consumido_hoy(_DIARIO)) == ""
    assert cdc.margen_del_dia({"kcal": 0}, cdc.consumido_hoy(_DIARIO)) == ""


def test_el_bloque_del_dia_lleva_el_margen_y_la_regla(monkeypatch):
    import coach_day_context as cdc
    monkeypatch.setattr(cdc, "metas_del_dia", lambda fd, plan: dict(_METAS))
    out = cdc.build_day_gap_context({}, None, _DIARIO, 22.5)
    assert "MARGEN QUE LE QUEDA HOY" in out and "grasas: quedan ~5 g" in out
    assert cdc.REGLA_MARGEN in out


@pytest.fixture
def diario(monkeypatch):
    import tools
    import db_facts
    import coach_day_context as cdc
    monkeypatch.setattr(tools, "user_tz_offset_min", lambda uid: 240)
    monkeypatch.setattr(tools, "_local_date_str_for_user", lambda uid=None: "2026-09-24")
    monkeypatch.setattr(db_facts, "get_consumed_meals_today", lambda uid, date_str=None, tz_offset_mins=None: _DIARIO)
    monkeypatch.setattr(tools, "_perfil_y_plan_de_hoy", lambda uid: ({}, None))
    monkeypatch.setattr(cdc, "metas_del_dia", lambda fd, plan: dict(_METAS))
    return tools


def test_la_nota_tras_registrar_trae_el_margen(diario):
    nota = diario._nota_total_del_dia("u1", 0)
    assert "TOTAL REAL DE HOY" in nota
    assert "grasas: quedan ~5 g" in nota and "Sin margen (o casi) en: grasas" in nota


def test_la_nota_de_otro_dia_no_lleva_margen(diario):
    assert "MARGEN" not in diario._nota_total_del_dia("u1", 1)


def test_el_margen_es_best_effort(diario, monkeypatch):
    def revienta(uid):
        raise RuntimeError("perfil caído")
    monkeypatch.setattr(diario, "_perfil_y_plan_de_hoy", revienta)
    nota = diario._nota_total_del_dia("u1", 0)
    assert "TOTAL REAL DE HOY" in nota and "MARGEN" not in nota


def test_la_regla_r2_va_en_el_prompt_compartido():
    from prompts import chat_agent
    r = chat_agent._CHAT_RESOLVE_RULES
    assert "R2. ¿ME LO COMO?" in r and "MARGEN QUE LE QUEDA HOY" in r and "grasas" in r


def test_marker():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"', _src("app.py"))
    assert m and int(m.group(1)) >= 226 and m.group(2) >= "2026-09-25"


def test_frontend_vuelve_al_chat_de_hoy():
    front = _BACKEND.parent / "frontend" / "src"
    if not (front / "__tests__" / "lote226.test.jsx").exists():
        pytest.skip("el frontend de este checkout no trae el lote 226")
    side = (front / "components" / "agent" / "SidebarRecientes.jsx").read_text(encoding="utf-8")
    assert "Volver al chat de hoy" in side
