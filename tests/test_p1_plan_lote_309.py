# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-309 · 2026-09-25] Lo COCIDO de un paso no se iguala a la base cruda/seca de la lista.

La lista compra en la base del catálogo (granos en seco, carnes en crudo); el sincronizador igualaba el número del paso
sin mirar la forma: «mide 25 g de quinoa cocida» con «30 g de quinoa» (seca) en la lista pasaba a «30 g de quinoa
cocida» (unos 10 g secos), y «desmenuza 150 g de pechuga cocida» con 200 g crudos, a «200 g de pechuga cocida»."""
from __future__ import annotations

import pathlib

import graph_orchestrator as go
import pasos_cantidades as pc

_BACKEND = pathlib.Path(__file__).resolve().parents[1]


def test_la_quinoa_cocida_del_paso_no_copia_los_gramos_secos_de_la_lista():
    m = {"ingredients": ["30 g de quinoa", "30 g de alcachofa", "1 diente de ajo"],
         "recipe": ["Mise en place: mide 25 g de quinoa cocida; corta 30 g de alcachofa."]}
    go._sync_recipe_step_quantities(m)
    assert "25 g de quinoa cocida" in m["recipe"][0], m["recipe"][0]


def test_la_pechuga_cocida_del_paso_no_copia_los_gramos_crudos():
    m = {"ingredients": ["200 g de pechuga de pollo", "1 cebolla"],
         "recipe": ["Montaje: desmenuza 150 g de pechuga de pollo cocida y mézclala con la cebolla."]}
    go._sync_recipe_step_quantities(m)
    assert "150 g de pechuga de pollo cocida" in m["recipe"][0], m["recipe"][0]


def test_la_clara_del_paso_no_copia_los_gramos_del_huevo_entero():
    # batería real (nocturno, 25-sep): «200 g de clara de huevo» se emparejaba por su 2.ª palabra con «60 g de huevo»
    m = {"ingredients": ["60 g de huevo", "6 claras de huevo", "1 rebanada de pan integral"],
         "recipe": ["Mise en place: mide 1 huevo y 200 g de clara de huevo."]}
    go._sync_recipe_step_quantities(m)
    assert "200 g de clara de huevo" in m["recipe"][0], m["recipe"][0]


def test_en_la_misma_forma_se_sincroniza_como_siempre():
    seca = {"ingredients": ["30 g de quinoa"], "recipe": ["Mise en place: mide 25 g de quinoa y enjuágala."]}
    go._sync_recipe_step_quantities(seca)
    assert "mide 30 g de quinoa" in seca["recipe"][0], seca["recipe"][0]
    cocida = {"ingredients": ["½ taza de arroz blanco cocido"], "recipe": ["Mise en place: mide 60 g de arroz blanco cocido."]}
    go._sync_recipe_step_quantities(cocida)
    assert "½ taza de arroz blanco cocido" in cocida["recipe"][0], cocida["recipe"][0]


def test_la_regla_sola():
    assert pc.otra_base("mide 25 g de quinoa cocida;", 5, 19, ["30 g de quinoa"], "quinoa") is True
    assert pc.otra_base("mide 25 g de quinoa y enjuágala", 5, 19, ["30 g de quinoa"], "quinoa") is False
    assert pc.otra_base("mide 25 g de quinoa cocida", 5, 19, ["30 g de arroz"], "quinoa") is False     # sin línea
    assert pc.otra_base(None, 0, 0, None, "x") is False


def test_ancla():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert "otra_base(step, mm.start(), mm.end(), ings, _ft):  # [P1-PLAN-LOTE-309]" in src
