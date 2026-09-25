# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-243 · 2026-09-25] El día determinista excluye por CLASE los rechazos y ve el texto libre.

Auditoría del 25-sep: el chip «Pescado» no excluía plantillas de sardinas, bacalao o atún (comparación por nombre exacto)
y lo escrito en «Otra alergia» no contaba cuando el perfil guardado traía chips.
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import deterministic_day as dd  # noqa: E402


def _n(xs):
    return {str(x).lower() for x in xs}


def test_el_rechazo_de_pescado_excluye_las_especies():
    _a, excl = dd.restricciones_del_formulario({"dislikes": ["Pescado"]}, {})
    assert {"sardina", "bacalao", "atun"} <= _n(excl), excl
    assert not ({"camaron", "camarones"} & _n(excl)), "los mariscos no son pescado (lote 210)"


def test_texto_libre_del_perfil_y_del_formulario():
    al, excl = dd.restricciones_del_formulario({"allergies": ["Mani"], "otherAllergies": "fresa"},
                                                {"dislikes": [], "otherDislikes": "hongos"})
    assert {"mani", "fresa"} <= _n(al), al
    assert {"champinones", "hongos"} <= _n(excl), excl


def test_el_centinela_no_es_un_alergeno():
    al, excl = dd.restricciones_del_formulario({"allergies": ["Ninguna"], "dislikes": ["Ninguno"]}, {})
    assert al == [] and excl == []


def test_las_plantillas_de_sardina_quedan_fuera():
    import dish_registry as dr
    _a, excl = dd.restricciones_del_formulario({"dislikes": ["Pescado"]}, {})
    t = {"constituents": [{"name": "Sardinas en lata", "canonical": "Sardinas"}, {"name": "Arroz"}]}
    try:
        from constants import pantry_names_match as pnm
    except Exception:
        pnm = None
    assert dr._template_uses_excluded_food(t, excl, pnm)


def test_cableado():
    src = (_BACKEND / "deterministic_day.py").read_text(encoding="utf-8")
    assert "_alergias_ft, _excl_ft = restricciones_del_formulario(_hp, _fd)" in src
    assert "tooltip-anchor: P1-PLAN-LOTE-243-DIA-DETERMINISTA" in src


def test_marker():
    import re
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"',
                  (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 243
