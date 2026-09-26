# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-338 · 2026-09-25] Bariátrica: la fruta deshidratada pasa a fresa.

Batería real del 25-sep (bariátrica + SOP): «55 g de dátiles» en la merienda PM → el revisor lo rechazó CRÍTICO (azúcar
concentrado → dumping), el reintento no convergió y el plan salió degradado tras 12,5 min."""
from __future__ import annotations

import condition_rules as cr
import graph_orchestrator as go

BARI = {"medicalConditions": ["Cirugía Bariátrica"]}


def test_las_sustituciones_bariatricas_traen_la_fruta_deshidratada():
    blob = " ".join(str(s) for s in cr.collect_substitutions(BARI)).lower()
    assert "dátiles" in blob and "pasas" in blob and "fresa" in blob
    dm2 = " ".join(str(s) for s in cr.collect_substitutions({"medicalConditions": ["Diabetes tipo 2"]})).lower()
    assert "dátiles" not in dm2                    # solo la bariátrica: el DM2 ya tiene su tope de fruta dulce (lote 182)


def test_los_datiles_de_la_merienda_pasan_a_fresa():
    plan = {"days": [{"day": 1, "meals": [{
        "meal": "Merienda PM", "name": "Yogurt griego con dátiles",
        "ingredients": ["120 g de yogurt griego sin azúcar", "55 g de dátiles"],
        "ingredients_raw": ["120 g de yogurt griego sin azúcar", "55 g de dátiles"],
        "recipe": ["Mise en place: pica 55 g de dátiles.", "Montaje: sirve el yogurt con los dátiles."]}]}]}
    assert go._apply_condition_substitutions(plan, BARI) >= 1
    ings = plan["days"][0]["meals"][0]["ingredients"]
    assert not any("dátil" in i.lower() for i in ings), ings
    assert any("fresa" in i.lower() and i.lstrip().startswith("55") for i in ings), ings


def test_el_prompt_lo_dice():
    assert "frutas DESHIDRATADAS (dátiles" in cr.build_condition_prompt(BARI)
