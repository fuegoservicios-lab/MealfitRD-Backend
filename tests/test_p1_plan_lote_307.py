# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-307 · 2026-09-25] El huevo tiene su propio detector numérico en el contrato de la forma.

Batería real del 25-sep (perfil del dueño): «bate 4 3 huevos y 1 clara de huevo con el comino». El tope diario de
enteros dejó «3 huevos» + «1 clara» en la lista; el contrato no veía «4 huevos» como mención (la cola «huevos con el
comino» nombra dos alimentos del catálogo y V7 la descarta) y su respaldo cambiaba sólo la palabra «huevos». En el mismo
plan, con la lista sólo de claras, el paso seguía diciendo «bate 1 huevo con ajo en polvo»."""
from __future__ import annotations

import copy
import pathlib

import recipe_contract as rc
from culinary_coherence import build_culinary_index

_BACKEND = pathlib.Path(__file__).resolve().parents[1]
_CAT = [
    {"name": "Huevo", "aliases": ["huevos", "huevos enteros"], "category": "Proteínas", "prep_methods": ["cocido"],
     "density_g_per_unit": 50},
    {"name": "Clara de huevo", "aliases": ["claras de huevo", "clara de huevo", "claras", "clara de huevos"],
     "category": "Proteínas", "prep_methods": ["cocido"], "density_g_per_unit": 33},
    {"name": "Yema de huevo", "aliases": ["yema de huevo", "yemas de huevo", "yemas"], "category": "Proteínas",
     "prep_methods": ["cocido"], "density_g_per_unit": 17},
    {"name": "Comino", "aliases": ["comino molido"], "category": "Especias", "prep_methods": ["crudo"]},
    {"name": "Ajo", "aliases": ["ajos"], "category": "Vegetales", "prep_methods": ["crudo", "sofrito"]},   # la cola de V7
    # son 4 palabras («huevo con ajo en»): en el catálogo real casa «Ajo», y con dos alimentos V7 descarta la mención
    {"name": "Ajo en polvo", "aliases": ["ajo en polvo"], "category": "Especias", "prep_methods": ["crudo"]},
    {"name": "Cebolla", "aliases": ["cebollas"], "category": "Vegetales", "prep_methods": ["crudo", "sofrito"]},
]
_INDEX = build_culinary_index(_CAT)


def _meal(ings, rec):
    return {"name": "Revoltillo", "meal": "Desayuno", "ingredients": list(ings), "ingredients_raw": list(ings),
            "recipe": list(rec)}


def test_el_numero_viejo_no_queda_delante_del_reparto():
    m = _meal(["3 huevos", "1 clara de huevo", "1 cdta de aceite de oliva"],
              ["Mise en place: bate 4 huevos con el comino.", "El Toque de Fuego: cuaja 2 min."])
    rc.reconcile_meal(m, _INDEX)
    assert m["recipe"][0] == "Mise en place: bate 3 huevos y 1 clara de huevo con el comino.", m["recipe"][0]


def test_si_el_paso_ya_nombra_la_clara_solo_cambia_el_conteo_de_enteros():
    m = _meal(["3 huevos", "1 clara de huevo"], ["Mise en place: bate 4 huevos y 1 clara con el comino."])
    rc.reconcile_meal(m, _INDEX)
    assert m["recipe"][0] == "Mise en place: bate 3 huevos y 1 clara con el comino.", m["recipe"][0]


def test_lista_solo_de_claras_con_otro_alimento_en_la_cola():
    m = _meal(["1 clara de huevo", "1 cebolla"], ["Mise en place: bate 1 huevo con ajo en polvo.",
                                                 "El Toque de Fuego: añade la clara y revuelve 2-3 min."])
    rc.reconcile_meal(m, _INDEX)
    assert m["recipe"][0] == "Mise en place: bate 1 clara de huevo con ajo en polvo.", m["recipe"][0]


def test_el_huevo_que_la_lista_da_en_gramos_tambien_cuenta():
    # batería real (nocturno, 25-sep): «60 g de huevo» + «6 claras» se leía «sólo claras» y «añade el huevo» → «añade las
    # claras y las claras»
    assert rc.egg_forms_in_list(["60 g de huevo", "6 claras de huevo"], _INDEX)["Huevo"] == 1.0
    m = _meal(["60 g de huevo", "6 claras de huevo"], ["Mise en place: mide 1 huevo y ten listas las claras.",
                                                      "El Toque de Fuego: añade el huevo y las claras y revuelve 3 min."])
    antes = copy.deepcopy(m)
    rc.egg_forms_step_sync(m, _INDEX)
    assert m == antes, m["recipe"]


def test_lo_que_ya_estaba_bien_sigue_igual_y_es_idempotente():
    bien = _meal(["3 huevos", "1 clara de huevo"], ["Mise en place: bate 3 huevos y 1 clara de huevo con el comino."])
    antes = copy.deepcopy(bien)
    assert rc.egg_forms_step_sync(bien, _INDEX)["reescritas"] == 0 and bien == antes
    m = _meal(["3 huevos", "1 clara de huevo"], ["Mise en place: bate los 4 huevos con el comino."])
    rc.reconcile_meal(m, _INDEX)
    uno = copy.deepcopy(m)
    rc.reconcile_meal(m, _INDEX)
    assert m == uno and m["recipe"][0] == "Mise en place: bate 3 huevos y 1 clara de huevo con el comino."
    codorniz = _meal(["3 huevos", "1 clara de huevo"], ["Mise en place: cuece 4 huevos de codorniz."])
    rc.egg_forms_step_sync(codorniz, _INDEX)
    assert "4 3" not in codorniz["recipe"][0] and "4 huevos de codorniz" in codorniz["recipe"][0]


def test_ancla():
    src = (_BACKEND / "recipe_contract.py").read_text(encoding="utf-8")
    assert "tooltip-anchor: P1-PLAN-LOTE-307-HUEVO-NUMERICO" in src and "for mm in _EGG_NUM_RE.finditer(paso):" in src
