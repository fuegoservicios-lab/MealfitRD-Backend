# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-260 · 2026-09-25] Un plan renal que un redondeo sube sobre el techo se vuelve a recortar antes de tirarlo.

Batería rd257: el plan de la IA (renal + gota) quedaba sobre el techo tras la cuantización, el truth-up y el rebalanceo;
los tres rechecks solo marcaban `meals_enforced=False` y el gate duro lo cambiaba por el plan de emergencia.
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

_TABLA = {"huevo": (6.3, 0.4, 4.8), "pechuga": (23.0, 0.0, 2.0), "arroz": (2.7, 28.0, 0.3)}


class _DB:
    def macros_from_ingredient_string(self, s):
        m = re.match(r"\s*([\d.]+)\s*(?:g de )?(.*)", str(s).lower())
        if not m:
            return None
        n, nombre = float(m.group(1)), m.group(2)
        for clave, (p, c, f) in _TABLA.items():
            if clave in nombre:
                k = n if clave == "huevo" else n / 100.0
                return {"protein": p * k, "carbs": c * k, "fats": f * k, "kcal": 4 * p * k + 4 * c * k + 9 * f * k}
        return None


def _plan(cap=40):
    return {"calories": 0, "renal_protein_cap": {"applied": True, "protein_g": cap, "meals_enforced": False},
            "days": [{"day": 1, "meals": [
                {"name": "Huevos", "protein": 19, "carbs": 1, "fats": 14, "ingredients": ["3 huevos"]},
                {"name": "Pollo y arroz", "protein": 32, "carbs": 42, "fats": 3,
                 "ingredients": ["125 g de pechuga de pollo", "150 g de arroz"]}]}]}


def test_reenforzar_devuelve_el_plan_al_techo_y_lo_marca():
    p = _plan()
    assert rr.reenforzar(p, 40, _DB()) is True
    assert p["renal_protein_cap"]["meals_enforced"] is True
    dia = sum(m["protein"] for m in p["days"][0]["meals"])
    assert dia <= 42, dia


def test_sin_techo_aplicado_no_hace_nada():
    p = _plan()
    p["renal_protein_cap"]["applied"] = False
    assert rr.reenforzar(p, 40, _DB()) is False


def test_los_tres_rechecks_lo_intentan_antes_de_escalar():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    for v in ("_rc_final_ok", "_rc_tu_ok", "_rc_reb_ok"):
        assert f'if not {v} and not __import__("recorte_renal").reenforzar(' in src, v
    assert "P1-PLAN-LOTE-260-REENFORZAR" in (_BACKEND / "recorte_renal.py").read_text(encoding="utf-8")


def test_marker():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"',
                  (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 260
