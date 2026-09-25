# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-238 · 2026-09-25] El relleno de ganar músculo respeta el país del mercado y las restricciones.

Auditoría del 25-sep: con «Nada» de tiempo el relleno era casabe en cualquier país (lote 221) y ni ese ni el de
arroz/batata miraban alergias, rechazos o dieta.
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import graph_orchestrator as go  # noqa: E402

_NUT = {"macros": {"protein_g": 135, "carbs_g": 334, "fats_g": 69}}   # 2497 kcal


def _meal(slot, name, cals, carbs, prot, fats, ingredients):
    ings = list(ingredients)
    return {"meal": slot, "name": name, "cals": cals, "carbs": carbs, "protein": prot, "fats": fats,
            "ingredients": ings, "ingredients_raw": list(ings),
            "recipe": ["Mise en place: prepara.", "El Toque de Fuego: cocina.", "Montaje: sirve."]}


def _dias():
    return [{"day": 1, "meals": [
        _meal("Almuerzo", "Wrap de pollo", 600, 60, 45, 15, ["150 g de pollo", "1 tortilla integral"]),
        _meal("Cena", "Ensalada de sardinas", 450, 30, 30, 12, ["1 lata de sardinas", "1 taza de lechuga"]),
    ]}]


def _relleno(monkeypatch, fd):
    monkeypatch.setattr(go, "GAINMUSCLE_DAY_KCAL_FLOOR_ENABLED", True)
    monkeypatch.setattr(go, "GAINMUSCLE_DAY_KCAL_FLOOR_PCT", 0.95)
    monkeypatch.setenv("MEALFIT_COUNTRY_SYSTEM", "true")
    days = _dias()
    go._repair_gainmuscle_day_kcal(days, _NUT, fd)
    return [i for m in days[0]["meals"] for i in m["ingredients"]], [s for m in days[0]["meals"] for s in m["recipe"]]


def test_republica_dominicana_sigue_con_casabe(monkeypatch):
    ings, _ = _relleno(monkeypatch, {"mainGoal": "gain_muscle", "cookingTime": "none", "country": "DO"})
    assert any(i.endswith("g de casabe") for i in ings), ings


def test_espana_recibe_pan_integral_no_casabe(monkeypatch):
    ings, pasos = _relleno(monkeypatch, {"mainGoal": "gain_muscle", "cookingTime": "none", "country": "ES"})
    assert any(i.endswith("g de pan integral") for i in ings) and not any("casabe" in i for i in ings), ings
    assert any(s.startswith("🍞 Acompaña con el pan integral") for s in pasos), pasos


def test_mexico_recibe_tortilla_de_maiz(monkeypatch):
    ings, _ = _relleno(monkeypatch, {"mainGoal": "gain_muscle", "cookingTime": "none", "country": "MX"})
    assert any(i.endswith("g de tortilla de maíz") for i in ings), ings


def test_celiaco_en_espana_no_recibe_pan(monkeypatch):
    ings, _ = _relleno(monkeypatch, {"mainGoal": "gain_muscle", "cookingTime": "none", "country": "ES",
                                     "allergies": ["Gluten"]})
    assert not any("pan integral" in i for i in ings), ings
    assert any(i.endswith("g de tortilla de maíz") for i in ings), ings


def test_con_tiempo_la_batata_rechazada_no_entra(monkeypatch):
    ings, _ = _relleno(monkeypatch, {"mainGoal": "gain_muscle", "cookingTime": "30min", "country": "DO",
                                     "dislikes": [], "otherDislikes": "batata"})
    assert not any("batata" in i for i in ings), ings


def test_marker():
    import re
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"',
                  (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 238
