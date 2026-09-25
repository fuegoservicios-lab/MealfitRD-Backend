# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-264 · 2026-09-25] Una nota del sistema no es un paso de cocina.

Batería final (DM2 + HTA, metformina + losartán): dos comidas decían «⚕️ Ajuste clínico: se sustituyó cundeamor
(hipoglucemiante: se suma al antidiabético) por una alternativa segura» y en sus ingredientes seguía «100g de cundeamor».
La guarda que añade a la lista el vegetal que nombran los pasos leyó la nota, encontró «cundeamor» (Vegetales, 17 kcal)
y lo devolvió — después de la sustitución y sin revisor que lo viera.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import graph_orchestrator as g  # noqa: E402

_NOTA_DM2 = ("⚕️ Ajuste clínico (condición médica): se sustituyó cundeamor (hipoglucemiante: se suma al antidiabético) "
             "por una alternativa segura (sin azúcar añadida / baja en sodio) para tu condición.")


def _catalogo(monkeypatch):
    import shopping_calculator as sc
    filas = [{"name": "Cundeamor", "category": "Vegetales", "kcal_per_100g": 17},
             {"name": "Alfalfa", "category": "Vegetales", "kcal_per_100g": 23},
             {"name": "Brócoli", "category": "Vegetales", "kcal_per_100g": 34}]
    monkeypatch.setattr(sc, "get_master_ingredients", lambda: filas)
    monkeypatch.setattr(sc, "normalize_name", lambda s: str(s).strip().lower())


def _dia(*pasos):
    return [{"meals": [{"name": "Wrap fresco de cebada con tayota", "ingredients": ["60 g de cebada cocida",
                                                                              "¼ taza de tayota"],
                        "recipe": ["Mise en place: corta la tayota.", "El Toque de Fuego: saltea la tayota 5 min.",
                                   "Montaje: sirve.", *pasos]}]}]


def test_la_nota_clinica_no_devuelve_el_cundeamor(monkeypatch):
    _catalogo(monkeypatch)
    dias = _dia(_NOTA_DM2)
    assert g._add_missing_recipe_step_vegetables(dias) == 0
    assert not any("cundeamor" in i.lower() for i in dias[0]["meals"][0]["ingredients"])


def test_la_nota_de_embarazo_no_anade_lo_que_prohibe(monkeypatch):
    _catalogo(monkeypatch)
    dias = _dia("🤰 Seguridad alimentaria (embarazo/lactancia): evita la alfalfa y los germinados crudos.")
    assert g._add_missing_recipe_step_vegetables(dias) == 0


def test_la_nota_de_alergia_tampoco(monkeypatch):
    _catalogo(monkeypatch)
    dias = _dia("🛡️ Sustitución por alergia declarada: se reemplazó brócoli por una alternativa segura.")
    assert g._add_missing_recipe_step_vegetables(dias) == 0


def test_un_paso_de_cocina_si_anade(monkeypatch):
    _catalogo(monkeypatch)
    dias = _dia("El Toque de Fuego: agrega el brócoli picado y saltea 3 minutos.", _NOTA_DM2)
    assert g._add_missing_recipe_step_vegetables(dias) == 1
    ings = " ".join(dias[0]["meals"][0]["ingredients"]).lower()
    assert "brócoli" in ings and "cundeamor" not in ings


def test_ancla():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    i = src.index("def _add_missing_recipe_step_vegetables(")
    j = src.index("\ndef ", i + 10)
    assert '__import__("recipe_contract")._es_nota(s)' in src[i:j]


def test_marker():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"',
                  (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 264
