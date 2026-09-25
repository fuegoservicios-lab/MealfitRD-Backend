# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-259 · 2026-09-25] El motor de macros no pasa el techo renal al escalar carbohidratos.

Traza dentro del proceso de la batería rd257 (renal + gota): el cierre final de banda del escudo llevó el día de 983 a
1.469 kcal escalando avena y arroz, y con ellos la proteína (60 → 73/87/70 g sobre un techo de 60).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import graph_orchestrator as go  # noqa: E402
import recorte_renal as rr  # noqa: E402

_TABLA = {"huevo": (6.3, 0.4, 4.8), "pechuga": (23.0, 0.0, 2.0), "avena": (5.0, 27.0, 2.5)}


class _DB:
    def macros_from_ingredient_string(self, s):
        m = re.match(r"\s*([\d.]+)\s*(?:g de |tazas? de )?(.*)", str(s).lower())
        if not m:
            return None
        n, nombre = float(m.group(1)), m.group(2)
        for clave, (p, c, f) in _TABLA.items():
            if clave in nombre:
                k = n if clave in ("huevo", "avena") else n / 100.0
                return {"protein": p * k, "carbs": c * k, "fats": f * k, "kcal": 4 * p * k + 4 * c * k + 9 * f * k}
        return None


def _dia():
    return [{"name": "Huevos y avena", "protein": 23, "carbs": 55, "fats": 12,
             "ingredients": ["2 huevos", "2 tazas de avena"]},
            {"name": "Pollo", "protein": 28, "carbs": 0, "fats": 2, "ingredients": ["120 g de pechuga de pollo"]}]


def test_con_techo_renal_el_dia_vuelve_al_techo():
    meals = _dia()
    plan = {"renal_protein_cap": {"applied": True, "protein_g": 40}}
    assert rr.retrim_dia(meals, plan, _DB())
    assert sum(m["protein"] for m in meals) <= 42
    assert meals[0]["ingredients"][1] == "2 tazas de avena"     # los carbohidratos (y sus kcal) no se tocan


def test_sin_techo_o_dentro_de_la_tolerancia_no_toca():
    assert not rr.retrim_dia(_dia(), {}, _DB())
    assert not rr.retrim_dia(_dia(), {"renal_protein_cap": {"applied": True, "protein_g": 50}}, _DB())


def test_nunca_peor_que_antes():
    """Si el día YA llegaba sobre el techo, el motor no recorta lo que no subió él (gate de 258-261)."""
    meals = _dia()                                   # 51 g
    plan = {"renal_protein_cap": {"applied": True, "protein_g": 40}}
    assert not rr.retrim_dia(meals, plan, _DB(), antes=51)
    assert meals[1]["ingredients"] == ["120 g de pechuga de pollo"]
    meals = _dia()
    assert rr.retrim_dia(meals, plan, _DB(), antes=40)          # el motor lo subió de 40 a 51: vuelve al techo
    assert sum(m["protein"] for m in meals) <= 42


def test_el_motor_lo_llama():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    i = src.index("def apply_update_macro_engine(")
    j = src.index("\ndef ", i + 10)
    assert '__import__("recorte_renal").retrim_dia(_meals, plan_data, db, antes=_p0_ume)' in src[i:j]
    assert "P1-PLAN-LOTE-259-MOTOR-RENAL" in (_BACKEND / "recorte_renal.py").read_text(encoding="utf-8")


def test_marker():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"',
                  (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 259
