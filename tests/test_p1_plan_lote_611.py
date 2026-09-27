# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-611 · 2026-09-27] El grano que pone el presupuesto cabe en el tiempo del formulario.

Corpus: las 27 sustituciones «quinoa → Arroz integral» cayeron todas en usuarios de «30 min» (o «Nada»); el paso seguía
con el hervor de la quinoa («unos 15 min») y una merienda dulce salía «…y arroz integral tostado».
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import graph_orchestrator as go  # noqa: E402
import presupuesto_tiempo as pt  # noqa: E402

_PRECIOS = {"quinoa": 300.0, "arroz integral": 70.0, "arroz blanco": 45.0, "avena": 60.0}

# Batería real (texto libre, maní/sésamo, 30 min, día 2) antes del ajuste de presupuesto.
_ALMUERZO = {
    "meal": "Almuerzo",
    "name": "Pollo criollo a la plancha con quinoa al cebollín y ensalada fresca de berro y repollo",
    "ingredients": ["1¼ pechugas de pollo (≈210 g)", "½ taza de quinoa", "1½ cucharadas de cebollín picado",
                    "1 diente de ajo", "1 taza de berro", "1 taza de repollo rallado", "1 limón",
                    "1 cda de aceite de oliva", "Sal al gusto"],
    "recipe": ["Mise en place: enjuaga ½ taza de quinoa (80 g); corta la pechuga en filetes de grosor parejo y pica el "
               "ajo y el cebollín.",
               "El Toque de Fuego: cocina la quinoa en agua según el paquete, unos 15 min; al final integra el cebollín. "
               "Cocina el pollo en sartén a fuego medio-alto, 5-7 min por lado, hasta 74 °C.",
               "Montaje: sirve la quinoa con el pollo y la ensalada."],
}
_MERIENDA = {
    "meal": "Merienda",
    "name": "Yogurt natural con fresas y quinoa tostada",
    "ingredients": ["¾ taza de yogurt natural sin azúcar", "80 g de fresas en trozos", "60 g de quinoa", "¼ cdta de miel"],
    "recipe": ["El Toque de Fuego: cocina la quinoa en agua hirviendo durante 12-15 minutos, hasta que esté tierna; "
               "escúrrela y tuéstala en una sartén seca 2-3 minutos.",
               "Montaje: sirve el yogurt en un tazón, añade las fresas y la quinoa tostada, y termina con la miel."],
}


def _form(tiempo="30min", condiciones=("Ninguna",), dislikes=()):
    return {"cookingTime": tiempo, "medicalConditions": list(condiciones), "country": "DO", "budget": "medium",
            "allergies": ["Maní", "Sésamo"], "dislikes": list(dislikes)}


@pytest.fixture(autouse=True)
def _precios(monkeypatch):
    monkeypatch.setattr(go, "_budget_build_master_price_map", lambda: {"catalogo": 1})

    def _precio(nombre, _mm):
        n = str(nombre).lower()
        return next((p for k, p in _PRECIOS.items() if k in n), 0.0)

    monkeypatch.setattr(go, "_budget_master_price_per_lb", _precio)


def _dias(*comidas):
    return [{"day": 1, "meals": [copy.deepcopy(c) for c in comidas]}]


def test_30_min_la_quinoa_pasa_a_arroz_blanco_con_su_hervor():
    dias = _dias(_ALMUERZO)
    assert go._apply_budget_cheapen_pass(dias, _form(), force=True) == 1
    m = dias[0]["meals"][0]
    assert m["_budget_substitutions"] == ["quinoa → Arroz blanco"]
    assert not any("integral" in x.lower() for x in m["ingredients"]), m["ingredients"]
    fuego = m["recipe"][1]
    assert "arroz blanco en agua según el paquete, unos 15-20 min" in fuego, fuego
    assert "5-7 min por lado" in fuego                    # la plancha del pollo no se toca


def test_30_min_con_diabetes_la_quinoa_se_queda():
    dias = _dias(_ALMUERZO)
    assert go._apply_budget_cheapen_pass(dias, _form(condiciones=["Diabetes T2"]), force=True) == 0
    assert "½ taza de quinoa" in dias[0]["meals"][0]["ingredients"]


def test_sop_hereda_la_regla_glucemica():
    dias = _dias(_ALMUERZO)
    assert go._apply_budget_cheapen_pass(dias, _form(condiciones=["SOP (PCOS)"]), force=True) == 0


def test_nada_de_tiempo_ningun_arroz_cabe():
    dias = _dias(_ALMUERZO)
    assert go._apply_budget_cheapen_pass(dias, _form(tiempo="none"), force=True) == 0


def test_una_hora_arroz_integral_con_su_hervor():
    dias = _dias(_ALMUERZO)
    assert go._apply_budget_cheapen_pass(dias, _form(tiempo="1hour"), force=True) == 1
    m = dias[0]["meals"][0]
    assert m["_budget_substitutions"] == ["quinoa → Arroz integral"]
    assert "unos 35-45 min" in m["recipe"][1], m["recipe"][1]


def test_la_quinoa_de_un_plato_dulce_no_se_vuelve_arroz():
    dias = _dias(_MERIENDA)
    assert go._apply_budget_cheapen_pass(dias, _form(tiempo="plenty"), force=True) == 0
    assert "60 g de quinoa" in dias[0]["meals"][0]["ingredients"]


def test_quien_rechaza_el_arroz_no_lo_recibe():
    dias = _dias(_ALMUERZO)
    assert go._apply_budget_cheapen_pass(dias, _form(dislikes=["Arroz"]), force=True) == 0


def test_el_pase_por_items_caros_aplica_la_misma_regla():
    lista = [{"name": "Quinoa", "estimated_cost_rd": 520.0}, {"name": "Pechuga de pollo", "estimated_cost_rd": 300.0}]
    dias = _dias(_ALMUERZO)
    assert go._apply_budget_driver_aware_pass(dias, _form(condiciones=["Diabetes T2"]), lista) == 0
    dias = _dias(_ALMUERZO)
    assert go._apply_budget_driver_aware_pass(dias, _form(), lista) == 1
    m = dias[0]["meals"][0]
    assert m["_budget_substitutions"] == ["quinoa → Arroz blanco"]
    assert "unos 15-20 min" in m["recipe"][1], m["recipe"][1]


@pytest.mark.parametrize("linea, esperado", [("25 g de harina de quinoa", "Arroz integral"),
                                             ("½ taza de quinoa", "Arroz blanco")])
def test_la_harina_no_se_hierve(linea, esperado):
    assert pt.candidato(linea, "Arroz integral", _form(), {"name": "Pan de batata", "ingredients": [linea]}) == esperado


def test_otros_candidatos_no_cambian():
    assert pt.candidato("100 g de salmón", "Filete de pescado blanco", _form(tiempo="none")) == "Filete de pescado blanco"
