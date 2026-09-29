# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-932 · 2026-09-29] En la compra única, «lo que aguanta el mes» se decide por el NOMBRE del candidato.

`compra_unica.candidatos_del_dia` (lote 521) filtra, desde el día 4 de una compra única sin congelador, las proteínas que
el cerrador puede añadir. Sus tests le pasaban NOMBRES; los tres cierres de proteína le pasan las tuplas del cerrador,
`(magrez, nombre, info)`, y el filtro evaluaba `f"100 g de {tupla}"`: «100 g de (0.236, 'Camarones',
NutritionInfo(name='Camarones', kcal=85.0, …))». Con ese texto sólo «aguanta» lo que lleva la palabra en el nombre.

Medido en el VPS con el plan real de la batería rdb528 (compra mensual, alergia al pescado, día 25): de 38 candidatos el
filtro dejaba UNO, «Guisantes secos». Llamado por el nombre deja 16 (huevos, habichuelas, lentejas, garbanzos, gandules,
quesos curados, soya texturizada…). De ahí los 15 platos con guisantes secos del corpus, y días al 65-79 % de su proteína:
el cerrador no tenía con qué cerrar un desayuno ni una merienda.
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import compra_unica as cu  # noqa: E402

_SINGLE = {"shopping": {"main_cycle_days": 30, "fresh_topup_days": None, "freezer_mode": "none",
                        "batch_cooking": "never"}, "diet": {"type": "balanced", "allergies": []}}
_FD = {"_plan_policy_effective": _SINGLE}


class _Info:
    """La ficha del catálogo tal como la lleva el cerrador: su texto trae «name='…'», kcal y macros."""

    def __init__(self, name, protein, kcal):
        self.name, self.protein, self.kcal = name, protein, kcal

    def __repr__(self):
        return f"NutritionInfo(name={self.name!r}, kcal={self.kcal}, protein={self.protein}, carbs=0.0, fats=0.5, source='usda')"


def _tupla(nombre, protein=20.0, kcal=100.0):
    return (protein / kcal, nombre, _Info(nombre, protein, kcal))


_NOMBRES = ["Pechuga de pollo", "Camarones", "Huevos", "Lentejas", "Habichuelas Rojas", "Garbanzos", "Guisantes secos",
            "Queso gouda", "Yogurt griego entero"]
_DURAN = {"Huevos", "Lentejas", "Habichuelas Rojas", "Garbanzos", "Guisantes secos", "Queso gouda"}


def test_las_tuplas_del_cerrador_se_filtran_por_su_nombre():
    cands = [_tupla(n) for n in _NOMBRES]
    out = cu.candidatos_del_dia(cands, {"day": 25}, _FD)
    assert {c[1] for c in out} == _DURAN, [c[1] for c in out]
    assert all(c in cands for c in out), "las mismas tuplas, sin copiar"
    assert [c[1] for c in out] == [n for n in _NOMBRES if n in _DURAN], "en su orden"


def test_con_nombres_sueltos_da_lo_mismo():
    assert set(cu.candidatos_del_dia(list(_NOMBRES), {"day": 25}, _FD)) == _DURAN


def test_con_fichas_sueltas_tambien():
    fichas = [_Info(n, 20.0, 100.0) for n in _NOMBRES]
    assert {f.name for f in cu.candidatos_del_dia(fichas, {"day": 25}, _FD)} == _DURAN


def test_el_nombre_del_candidato():
    assert cu.nombre_del_candidato(_tupla("Lentejas")) == "Lentejas"
    assert cu.nombre_del_candidato(_Info("Huevos", 12.6, 139.0)) == "Huevos"
    assert cu.nombre_del_candidato("Garbanzos") == "Garbanzos"
    assert cu.nombre_del_candidato((_Info("Queso gouda", 24.9, 356.0), "queso gouda")) == "Queso gouda", "el par del pool"
    assert cu.nombre_del_candidato(None) == ""


def test_lo_de_siempre_sigue_igual():
    cands = [_tupla(n) for n in _NOMBRES]
    assert cu.candidatos_del_dia(cands, {"day": 2}, _FD) == cands, "antes del día 4"
    assert cu.candidatos_del_dia(cands, {"day": 25}, {}) == cands, "sin política de compra única"
    solo = [_tupla("Pechuga de pollo")]
    assert cu.candidatos_del_dia(solo, {"day": 25}, _FD) == solo, "si nada aguanta, la lista de siempre"


def test_con_el_knob_apagado_el_filtro_de_antes(monkeypatch):
    monkeypatch.setenv("MEALFIT_SINGLE_TRIP_CANDIDATE_BY_NAME", "false")
    out = cu.candidatos_del_dia([_tupla(n) for n in _NOMBRES], {"day": 25}, _FD)
    assert [c[1] for c in out] == ["Guisantes secos"], [c[1] for c in out]


def test_ancla():
    src = (_BACKEND / "compra_unica.py").read_text(encoding="utf-8")
    cuerpo = src[src.index("def candidatos_del_dia("):src.index("def _redondea(")]
    assert "nombre_del_candidato(c)" in cuerpo and "tooltip-anchor: P1-PLAN-LOTE-932" in src
    assert b"\x08" not in (_BACKEND / "compra_unica.py").read_bytes()
