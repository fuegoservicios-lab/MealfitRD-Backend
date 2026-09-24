# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-187 · 2026-09-23] Embarazo y lactancia: el pescado, ≤340 g por semana, sustituido (no recortado) por una
proteína que ese día no se repita. Batería real: «545 g de pescado en 3 días» (rd5 → emergencia), «460 g» (rd15)."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import embarazo_pescado as ep  # noqa: E402

_PREG = {"medicalConditions": ["Embarazo"], "gender": "female", "dietType": "balanced"}


class _DB:
    def grams_from_ingredient_string(self, s):
        s = str(s)
        m = re.search(r"\(≈\s*(\d+)\s*g\)", s) or re.match(r"\s*(\d+)\s*g\b", s)
        return float(m.group(1)) if m else 100.0

    def macros_from_ingredient_string(self, s):
        g = self.grams_from_ingredient_string(s)
        return {"grams": g, "kcal": 1.2 * g, "protein": 0.25 * g, "carbs": 0.0, "fats": 0.03 * g}

    def __getattr__(self, _n):
        return lambda *a, **k: None


def _meal(slot, nombre, lineas, pasos=None):
    return {"meal": slot, "name": nombre, "ingredients": list(lineas), "ingredients_raw": list(lineas),
            "recipe": list(pasos or ["Cocina y sirve."])}


def _plan():
    return {"days": [
        {"day": 1, "meals": [_meal("Almuerzo", "Tilapia al horno con plátano", ["120 g de filete de tilapia", "1 plátano verde"])]},
        {"day": 2, "meals": [_meal("Almuerzo", "Ensalada de atún claro", ["1 lata de atún claro en agua (≈120 g)", "1 tomate"])]},
        {"day": 3, "meals": [
            _meal("Desayuno", "Revoltillo de huevos", ["2 huevos", "½ tomate"]),
            _meal("Almuerzo", "Pollo guisado con arroz", ["150 g de pechuga de pollo", "½ taza de arroz"]),
            _meal("Cena", "Tilapia a la plancha con ensalada", ["220 g de filete de tilapia", "1 taza de lechuga"],
                  ["Sazona la tilapia con limón.", "Cocina la tilapia a la plancha 4 minutos por lado."]),
        ]},
    ]}


def _pescado_total(plan, db=_DB()):
    return sum(db.grams_from_ingredient_string(x) for d in plan["days"] for m in d["meals"]
               for x in m["ingredients"] if ep._PESCADO.search(x))


def test_el_pescado_de_mas_se_cambia_por_una_proteina_que_el_dia_no_repite():
    plan = _plan()
    assert _pescado_total(plan) == 460
    assert ep.limitar_pescado(plan, _PREG, db=_DB()) == 1
    cena = plan["days"][2]["meals"][2]
    assert cena["ingredients"][0] == "220 g de pechuga de pavo", "el pollo ya está en el almuerzo del día 3"
    assert cena["ingredients_raw"][0] == "220 g de pechuga de pavo", "la compra sigue a la lista"
    assert "tilapia" not in cena["name"].lower() and "pavo" in cena["name"].lower(), cena["name"]
    assert not any("tilapia" in p.lower() for p in cena["recipe"]), cena["recipe"]
    assert _pescado_total(plan) == 240 <= 340
    assert plan["days"][0]["meals"][0]["ingredients"][0] == "120 g de filete de tilapia", "lo que cabe no se toca"
    assert ep.limitar_pescado(plan, _PREG, db=_DB()) == 0, "idempotente"


def test_rechazo_declarado_salta_al_siguiente_sustituto():
    plan = _plan()
    ep.limitar_pescado(plan, dict(_PREG, dislikes=["Pavo"]), db=_DB())
    assert plan["days"][2]["meals"][2]["ingredients"][0] == "220 g de carne de res"


@pytest.mark.parametrize("form", [
    {"medicalConditions": ["Ninguna"], "dietType": "balanced"},              # sin embarazo
    dict(_PREG, dietType="pescatarian"),                                     # pescetariana: no hay a qué cambiar
])
def test_fuera_de_su_alcance_no_toca(form):
    plan = _plan()
    assert ep.limitar_pescado(plan, form, db=_DB()) == 0 and _pescado_total(plan) == 460


def test_knob_a_cero_apaga(monkeypatch):
    monkeypatch.setenv("MEALFIT_PREGNANCY_FISH_CAP_G", "0")
    plan = _plan()
    assert ep.limitar_pescado(plan, _PREG, db=_DB()) == 0


def test_el_adjetivo_dorado_no_es_pescado():
    assert not ep._PESCADO.search("½ plátano dorado") and not ep._PESCADO.search("1 pollo dorado")
    assert ep._PESCADO.search("1 filete de dorado")


def test_corre_en_la_sustitucion_clinica():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert '__import__("embarazo_pescado").limitar_pescado(plan, form_data)' in src


def test_knob_documentado():
    doc = (_BACKEND / "docs" / "knobs_reference.md").read_text(encoding="utf-8")
    assert "| `MEALFIT_PREGNANCY_FISH_CAP_G` | `340` |" in doc


def test_marker():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"',
                  (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 187 and m.group(2) >= "2026-09-23"
