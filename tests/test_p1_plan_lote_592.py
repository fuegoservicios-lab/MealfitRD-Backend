# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-592 · 2026-09-27] Cuando los topes atan la proteína del día, entra una proteína NUEVA.

Batería real (texto libre, maní/sésamo, día 3): 1.885 kcal y 129 g de proteína de 2.350/176 — huevos, claras, edamame y
cottage topados; el cierre final los subía y los topes los devolvían. Replay: +140 g de pechuga de pollo al guiso de la
cena → 161 g (piso 158).
"""
from __future__ import annotations

import copy
import re
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import graph_orchestrator as go  # noqa: E402
import proteina_nueva as pn  # noqa: E402

_TABLA = {"pechuga de pollo": (120, 23.0), "pechuga de pavo": (110, 24.0), "huevo": (150, 13.0),
          "edamame": (120, 11.0), "quinoa": (370, 14.0), "arepitas": (200, 4.0), "casabe": (330, 1.0)}


class _Info:
    def __init__(self, name, kcal, protein):
        self.name, self.kcal, self.protein, self.carbs, self.fats = name, kcal, protein, 0.0, 2.0


class _DB:
    def macros_from_ingredient_string(self, s):
        m = re.match(r"^\s*(\d+)\s*g\s+de\s+(.+)$", str(s))
        if not m:
            return None
        g, food = float(m.group(1)), m.group(2).lower()
        for k, (kc, p) in _TABLA.items():
            if k in food:
                return {"kcal": kc * g / 100, "protein": p * g / 100, "carbs": 0.0, "fats": 2.0 * g / 100}
        return None

    def lookup(self, s):
        return None


def _plan(almuerzo="Papa majada con alcachofa y edamame", cena="Quinoa guisada con lentejas y auyama",
          lineas_almuerzo=("300 g de edamame",), renal=False, kcal_dia=1885):
    def meal(slot, name, lineas, p, k):
        return {"meal": slot, "name": name, "ingredients": list(lineas), "ingredients_raw": list(lineas),
                "recipe": ["Mise en place: corta todo.", "El Toque de Fuego: guisa todo 10 min.", "Montaje: sirve."],
                "protein": p, "cals": k, "carbs": 100, "fats": 20}
    plan = {"calories": 2350, "macros": {"protein": "176g", "carbs": "235g", "fats": "78g"},
            "days": [{"day": 1, "meals": [
                meal("Desayuno", "Mangú con huevo cocido", ["3 huevos"], 43, 500),
                meal("Almuerzo", almuerzo, lineas_almuerzo, 47, 600),
                meal("Merienda", "Frutas picadas con cottage", ["120 g de queso cottage"], 18, 185),
                meal("Cena", cena, ["150 g de quinoa"], 21, kcal_dia - 1285)]}]}
    if renal:
        plan["renal_protein_cap"] = {"applied": True, "protein_g": 60}
    return plan


_FORM = {"dietType": "balanced", "cookingTime": "30min", "groceryDuration": "weekly", "country": "DO"}


@pytest.fixture(autouse=True)
def _candidatas(monkeypatch):
    cands = [(0.19, "Pechuga de pollo", _Info("Pechuga de pollo", 120, 23.0)),
             (0.22, "Pechuga de pavo", _Info("Pechuga de pavo", 110, 24.0)),
             (0.10, "Edamame", _Info("Edamame", 120, 11.0))]
    monkeypatch.setattr(go, "_safe_high_density_proteins", lambda *a, **k: list(cands))


def test_el_dia_corto_recibe_pollo_en_la_cena():
    plan = _plan()
    hechos = pn.cerrar(plan, _FORM, db=_DB())
    assert hechos and "pechuga de pollo" in hechos[0], hechos
    cena = plan["days"][0]["meals"][3]
    assert any(x.endswith("g de pechuga de pollo") for x in cena["ingredients"])
    assert any(x.endswith("g de pechuga de pollo") for x in cena["ingredients_raw"])
    assert any("pechuga de pollo" in p.lower() for p in cena["recipe"]), cena["recipe"]
    assert sum(float(m["protein"]) for m in plan["days"][0]["meals"]) >= 176 * 0.9 - 0.5


def test_si_el_dia_ya_usa_pollo_entra_otra_proteina():
    plan = _plan(almuerzo="Pollo guisado con papa", lineas_almuerzo=("150 g de pechuga de pollo",))
    hechos = pn.cerrar(plan, _FORM, db=_DB())
    assert hechos and "pavo" in hechos[0], hechos


@pytest.mark.parametrize("kw", [{"renal": True}, {"kcal_dia": 2520}])
def test_con_techo_renal_o_sin_sitio_en_el_dia_no_entra(kw):
    plan = _plan(**kw)
    antes = copy.deepcopy(plan)
    assert pn.cerrar(plan, _FORM, db=_DB()) == [] and plan == antes


def test_nunca_en_un_batido(monkeypatch):
    plan = _plan(almuerzo="Batido de lechosa con avena", cena="Batido de guineo y maní")
    antes = copy.deepcopy(plan)
    assert pn.cerrar(plan, _FORM, db=_DB()) == [] and plan == antes


def test_con_la_nevera_exigida_solo_entra_lo_que_la_nevera_tiene(monkeypatch):
    import nevera_exigida as ne
    monkeypatch.setattr(ne, "lista", lambda fd=None: ["pechuga de pavo", "huevos"])
    monkeypatch.setattr(ne, "admite", lambda nombre, fd=None: "pavo" in str(nombre).lower())
    plan = _plan()
    hechos = pn.cerrar(plan, _FORM, db=_DB())
    assert hechos and "pavo" in hechos[0], hechos
    monkeypatch.setattr(ne, "admite", lambda nombre, fd=None: False)
    plan = _plan()
    antes = copy.deepcopy(plan)
    assert pn.cerrar(plan, _FORM, db=_DB()) == [] and plan == antes


def test_knob_apagado(monkeypatch):
    monkeypatch.setenv("MEALFIT_PROTEIN_NEW_WHEN_CAPPED", "false")
    plan = _plan()
    assert pn.cerrar(plan, _FORM, db=_DB()) == []


def test_anclas():
    src = (_BACKEND / "protein_floor_last_word.py").read_text(encoding="utf-8")
    assert '__import__("proteina_nueva").cerrar(plan_data, form_data)' in src and "P1-PLAN-LOTE-592" in src
    src = (_BACKEND / "db_plans.py").read_text(encoding="utf-8")
    assert '_pflw_ins(_pd, form_data=locals().get("_clin_ctx") or None,' in src
