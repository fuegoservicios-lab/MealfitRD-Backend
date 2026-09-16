# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-70 · 2026-09-16] El suelo nunca dropea el ingrediente que da NOMBRE al plato.

Medido reproduciendo el escudo pre-INSERT sobre un plan REAL del 16-sep: para quitar una línea sub-servible, el
suelo quitó «10 g de habas cocidas» de «Mango Fresco con Nueces Mixtas y Habas». El plato conservó el nombre y
perdió el ingrediente. El repo ya tenía la regla para la proteína protagonista («JAMÁS drop — es la identidad del
plato»); esto la extiende a cualquier alimento que nombre el plato: sin headroom se sube al piso igual.
"""
from __future__ import annotations

import re
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]


def _src(p: Path) -> str:
    return p.read_text(encoding="utf-8")


class _DbFalso:
    def macros_from_ingredient_string(self, s):
        m = re.match(r"^\s*(\d+(?:[.,]\d+)?)\s*g\b", str(s))
        if not m:
            return None
        g = float(m.group(1).replace(",", "."))
        return {"kcal": g * 4.0, "protein": 0.0, "carbs": g * 0.5, "fats": 0.0}


def _dia(nombre, linea_corta, extras=("120 g de mango en cubos", "20 g de semillas de girasol")):
    ings = [linea_corta, *extras]
    return {"day": 1, "meals": [{"meal": "Merienda", "name": nombre, "cals": 2000,
                                 "protein": 10, "carbs": 40, "fats": 10,
                                 "ingredients": list(ings), "ingredients_raw": list(ings),
                                 "recipe": ["Mise en place: corta.", "Montaje: sirve."]}]}


def _lineas(d):
    return [str(x) for x in d["meals"][0]["ingredients"]]


# ─────────────────────────────── el alimento que nombra el plato

def test_el_ingrediente_del_nombre_se_sube_en_vez_de_dropearse():
    from graph_orchestrator import _floor_subservible_portions, PORTION_SHRINK_FLOOR_G
    d = _dia("Mango Fresco con Nueces Mixtas y Habas", "10 g de habas cocidas")
    _floor_subservible_portions([d], day_kcal_target=2000, db=_DbFalso())   # headroom 0: la comida agota el día
    lineas = _lineas(d)
    habas = [x for x in lineas if "habas" in x.lower()]
    assert habas, f"la identidad del plato se dropeó: {lineas}"
    assert float(re.match(r"^\s*(\d+(?:[.,]\d+)?)", habas[0]).group(1).replace(",", ".")) >= float(PORTION_SHRINK_FLOOR_G)


def test_un_alimento_que_no_nombra_el_plato_sigue_pudiendo_dropearse():
    """La regla es de identidad, no una amnistía general: sin headroom, lo demás sigue cayendo."""
    from graph_orchestrator import _floor_subservible_portions
    d = _dia("Mango Fresco con Nueces Mixtas", "10 g de habas cocidas")
    _floor_subservible_portions([d], day_kcal_target=2000, db=_DbFalso())
    assert not [x for x in _lineas(d) if "habas" in x.lower()], _lineas(d)


def test_el_helper_no_confunde_estado_ni_corte_con_identidad():
    from constants import strip_accents
    from graph_orchestrator import _linea_da_nombre_al_plato as f
    plato = {"name": "Mango Fresco con Nueces Mixtas y Habas"}
    assert f("10 g de habas cocidas", plato, strip_accents) is True
    assert f("120 g de mango en cubos", plato, strip_accents) is True
    assert f("20 g de semillas de girasol", plato, strip_accents) is False
    assert f("5 g de queso fresco", {"name": "Mango Fresco con Limón"}, strip_accents) is False, \
        "«fresco» es estado, no identidad"
    assert f("10 g de habas", {"name": ""}, strip_accents) is False


def test_el_ancla_sigue_en_el_fuente():
    src = _src(_BACKEND / "graph_orchestrator.py")
    assert "P1-PLAN-LOTE-70-NO-DROPEAR-LA-IDENTIDAD" in src
    i_id = src.index("if _linea_da_nombre_al_plato(s, meal, _sa):")
    i_drop = src.index("_drop_idx.append(idx)", i_id)
    assert i_id < i_drop, "la guarda de identidad debe decidirse ANTES de apuntar el drop"


# ─────────────────────────────── docs y marker

def test_docs_y_marker():
    doc = _src(_BACKEND / "docs" / "culinary_coherence.md")
    assert "P1-PLAN-LOTE-70" in doc and "identidad" in doc.lower()
    m = re.search(r'^_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"', _src(_BACKEND / "app.py"), re.M)
    assert m and int(m.group(1)) >= 70 and m.group(2) >= "2026-09-16"
