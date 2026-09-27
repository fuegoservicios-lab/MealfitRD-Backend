# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-584 · 2026-09-27] Lo que el nombre promete y los pasos cocinan vuelve a la lista.

Batería real (adulto mayor con HTA): «Pollo jugoso con majado de yautía y aguacate al cilantro» pelaba, cortaba y hervía
250 g de yautía que la lista no traía: ni se compraba ni contaba en las calorías.
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

import identidad_plato as ip  # noqa: E402
from culinary_coherence import build_culinary_index  # noqa: E402

_FILAS = [
    ("Yautía", ["yautia"], "Víveres", 112, 1.5, 27, 0.2),
    ("Mapuey", [], "Víveres", 100, 1.5, 24, 0.1),
    ("Pechuga de pollo", ["pechuga"], "Proteínas", 120, 22.5, 0, 2.6),
    ("Filete de pescado", ["pescado"], "Proteínas", 96, 20, 0, 1.7),
    ("Tilapia", [], "Proteínas", 96, 20, 0, 1.7),
    ("Aguacate", [], "Frutas", 160, 2, 9, 15),
    ("Cilantro", [], "Vegetales", 23, 2, 4, 0.5),
    ("Ajo", [], "Vegetales", 149, 6, 33, 0.5),
    ("Cebolla roja", ["cebolla"], "Vegetales", 40, 1, 9, 0.1),
    ("Limón", ["limon"], "Frutas", 29, 1, 9, 0.3),
    ("Aceite de oliva", [], "Grasas", 884, 0, 0, 100),
    ("Harina de trigo", [], "Despensa", 364, 10, 76, 1),
]
_CAT = [{"name": n, "aliases": a, "category": c, "ready_to_eat": False, "prep_methods": ["hervir"]}
        for n, a, c, *_ in _FILAS]


def _sa(s):
    return (s or "").lower().translate(str.maketrans("áéíóúñ", "aeioun"))


class _Info:
    def __init__(self, protein):
        self.protein = protein


class _DB:
    def _row(self, s):
        n = _sa(s)
        return next((r for r in sorted(_FILAS, key=lambda r: -len(r[0])) if _sa(r[0]) in n or any(a in n for a in r[1])),
                    None)

    def lookup(self, s):
        r = self._row(s)
        return _Info(r[4]) if r else None

    def category_of(self, s):
        r = self._row(s)
        return r[2] if r else None

    def grams_from_ingredient_string(self, s):
        m = re.match(r"\s*(\d+(?:\.\d+)?)\s*g", s)
        return float(m.group(1)) if m else None

    def macros_from_ingredient_string(self, s):
        g, r = self.grams_from_ingredient_string(s), self._row(s)
        if not (g is not None and r):
            return None
        f = g / 100.0
        return {"kcal": r[3] * f, "protein": r[4] * f, "carbs": r[5] * f, "fats": r[6] * f}


@pytest.fixture(scope="module")
def index():
    return build_culinary_index(_CAT)


_YAUTIA = {
    "name": "Pollo jugoso con majado de yautía y aguacate al cilantro",
    "ingredients": ["¾ pechuga de pollo (≈142 g)", "½ aguacate", "½ diente de ajo picado", "1 cda de cilantro picado",
                    "1 cdta de jugo de limón", "¾ cdta de aceite de oliva", "½ cebolla roja picada"],
    "ingredients_raw": ["150 g de pechuga de pollo", "0.5 aguacate", "0.5 diente de ajo picado", "1 cda de cilantro picado",
                        "1 cdta de jugo de limón", "¾ cdta de aceite de oliva", "0.5 cebolla roja picada"],
    "recipe": [
        "Mise en place: corta 142 g de pechuga de pollo en una pieza uniforme; pela y corta 250 g de yautía en cubos; pica "
        "½ diente de ajo, 1 cda de cilantro y ½ cebolla roja.",
        "El Toque de Fuego: hierve la yautía en agua hasta que el cuchillo entre sin fuerza, aproximadamente 15-20 minutos, "
        "y escúrrela bien antes de majarla. Cocina la pechuga con el ajo y la cebolla en una sartén antiadherente.",
        "Montaje: sirve el pollo sobre el majado de yautía y acompaña con el aguacate.",
    ],
}


def _correr(meal, index, margen, allergies=()):
    return ip._subir_identidad_del_modelo(meal, index, db=_DB(), allergies=list(allergies), margen=margen)


def test_la_yautia_del_nombre_vuelve_a_la_lista_y_a_la_compra(index):
    m = copy.deepcopy(_YAUTIA)
    margen = {"kcal": 300.0, "grasa": 20.0}
    hechos = _correr(m, index, margen)
    assert "+60 g de Yautía" in hechos, hechos
    assert "60 g de Yautía" in m["ingredients"] and "60 g de Yautía" in m["ingredients_raw"]
    assert margen["kcal"] == pytest.approx(300.0 - 112 * 0.6)


def test_sin_sitio_en_el_dia_no_entra(index):
    m = copy.deepcopy(_YAUTIA)
    assert not any("Yautía" in h for h in _correr(m, index, {"kcal": 20.0, "grasa": 20.0}))
    assert not any("autía" in x for x in m["ingredients"])


def test_lo_que_cabe_desde_la_mitad_del_piso(index):
    m = copy.deepcopy(_YAUTIA)
    hechos = _correr(m, index, {"kcal": 50.0, "grasa": 20.0})
    assert "+44 g de Yautía" in hechos, hechos


def test_un_dia_pasado_de_grasa_no_frena_lo_que_no_la_trae(index):
    m = copy.deepcopy(_YAUTIA)
    assert "+60 g de Yautía" in _correr(m, index, {"kcal": 300.0, "grasa": -2.9})   # batería real: día 1 a 68/65 g
    m = copy.deepcopy(_YAUTIA)
    m["ingredients"] = [x for x in m["ingredients"] if "aguacate" not in x]
    m["ingredients_raw"] = [x for x in m["ingredients_raw"] if "aguacate" not in x]
    hechos = _correr(m, index, {"kcal": 300.0, "grasa": -2.9})
    assert not any("Aguacate" in h for h in hechos), "el aguacate sí trae grasa: con el día pasado no entra"


def test_alergia_o_sustitucion_no_vuelven(index):
    m = copy.deepcopy(_YAUTIA)
    assert not any("Yautía" in h for h in _correr(m, index, {"kcal": 300.0, "grasa": 20.0}, allergies=["yautía"]))
    m = copy.deepcopy(_YAUTIA)
    m["_fresh_substituted"] = ["60 g de yautía → mapuey"]
    assert not any("Yautía" in h for h in _correr(m, index, {"kcal": 300.0, "grasa": 20.0}))


def test_sin_nombrarla_el_nombre_no_se_anade(index):
    m = copy.deepcopy(_YAUTIA)
    m["name"] = "Pollo jugoso con aguacate al cilantro"
    assert not any("Yautía" in h for h in _correr(m, index, {"kcal": 300.0, "grasa": 20.0}))


def test_la_especie_del_paso_es_la_proteina_de_la_lista(index):
    m = {"name": "Tilapia a la plancha con mapuey",
         "ingredients": ["1¾ filetes de pescado (≈250 g)", "350 g de mapuey", "½ cda de aceite de oliva"],
         "ingredients_raw": ["250 g de filete de pescado", "350 g de mapuey", "½ cda de aceite de oliva"],
         "recipe": ["Mise en place: pela y corta 350 g de mapuey en trozos; corta 250 g de tilapia en porciones.",
                    "El Toque de Fuego: hierve el mapuey 20 min y cocina la tilapia a la plancha 4 min por lado.",
                    "Montaje: sirve la tilapia con el mapuey."]}
    antes = list(m["ingredients"])
    _correr(m, index, {"kcal": 300.0, "grasa": 20.0})
    assert m["ingredients"] == antes


def test_knob_apagado(index, monkeypatch):
    monkeypatch.setenv("MEALFIT_DISH_IDENTITY_MISSING", "false")
    m = copy.deepcopy(_YAUTIA)
    assert not any("Yautía" in h for h in _correr(m, index, {"kcal": 300.0, "grasa": 20.0}))


def test_con_alergia_declarada_el_escaner_decide(index):
    m = {"name": "Arepitas de trigo con queso", "ingredients": ["½ cebolla roja picada"],
         "ingredients_raw": ["0.5 cebolla roja picada"],
         "recipe": ["Mise en place: mide 60 g de harina de trigo y pica la cebolla roja.",
                    "El Toque de Fuego: mezcla la harina de trigo con agua, forma arepitas y dóralas 3 min por lado.",
                    "Montaje: sirve las arepitas."]}
    assert not any("Harina" in h for h in _correr(copy.deepcopy(m), index, {"kcal": 300.0, "grasa": 20.0},
                                                   allergies=["Gluten"])), "«Gluten» no es una palabra de la harina"
    assert "+25 g de Harina de trigo" in _correr(copy.deepcopy(m), index, {"kcal": 300.0, "grasa": 20.0})
