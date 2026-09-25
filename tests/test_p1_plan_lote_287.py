# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-287 · 2026-09-25] Lo que el usuario lee tras un cambio por presupuesto.

Plan del dueño («Económico»): el aviso repetía «yogurt griego → Yogurt natural» tres veces, el título decía «Arroz frío
con Filete de pescado blanco estilo ceviche» y una línea «1 cda de maní fileteado» (el corte de las almendras)."""
from __future__ import annotations

import pathlib

import pytest

import presupuesto_texto as pt

_BACKEND = pathlib.Path(__file__).resolve().parents[1]


def test_titulo_con_minuscula_a_media_frase():
    assert pt.sustituir("Arroz frío con salmón estilo ceviche", r"salm[oó]n", "Filete de pescado blanco",
                        titulo=True) == "Arroz frío con filete de pescado blanco estilo ceviche"
    assert pt.sustituir("Salmón a la plancha", r"salm[oó]n", "Filete de pescado blanco",
                        titulo=True) == "Filete de pescado blanco a la plancha"
    assert pt.sustituir("Yuca con Habas Guisadas", r"\bhabas?\b", "Habichuelas rojas",
                        titulo=True) == "Yuca con Habichuelas rojas Guisadas"     # título en mayúsculas: se copia
    # las líneas conservan el nombre del catálogo (el pulido final las baja)
    assert pt.sustituir("150 g de salmón", r"salm[oó]n", "Filete de pescado blanco") == "150 g de Filete de pescado blanco"


def test_el_corte_del_premium_no_sobrevive_y_la_unidad_no_se_duplica():
    assert pt.sustituir("1 cda de almendras fileteadas", r"almendras?", "Maní") == "1 cda de Maní picado"
    assert pt.sustituir("1 cda de maní fileteado", r"nueces|nuez", "Maní") == "1 cda de maní fileteado"   # sin cambio: no casó
    assert pt.sustituir("½ filete de mero", r"mero", "Filete de pescado blanco") == "½ filete de pescado blanco"


def test_el_aviso_no_repite():
    dias = [{"meals": [{"_budget_substitutions": ["yogurt griego → Yogurt natural", "salmón → Filete de pescado blanco"]},
                       {"_budget_substitutions": ["yogurt griego → Yogurt natural"]}]},
            {"meals": [{"_budget_substitutions": ["Yogurt griego → Yogurt natural", "kale → Espinacas"]}, {}]}]
    assert pt.sustituciones_unicas(dias) == ["yogurt griego → Yogurt natural", "salmón → Filete de pescado blanco",
                                             "kale → Espinacas"]
    assert len(pt.sustituciones_unicas(dias * 5, limite=2)) == 2


@pytest.fixture()
def cheapen_env(monkeypatch):
    import graph_orchestrator as go
    monkeypatch.setattr(go, "BUDGET_CHEAPEN_PASS_ENABLED", True)
    monkeypatch.setattr(go, "BUDGET_CHEAPEN_MAX_SUBS", 4)
    monkeypatch.setattr(go, "_budget_build_master_price_map",
                        lambda: {"salmon": 600.0, "filete de pescado blanco": 127.0, "almendras": 700.0, "mani": 150.0})
    return go


def test_el_pase_del_formulario_escribe_asi(cheapen_env):
    go = cheapen_env
    dias = [{"day": 1, "meals": [{
        "meal": "Almuerzo", "name": "Arroz frío con salmón estilo ceviche",
        "ingredients": ["150g de salmón", "1 cda de almendras fileteadas", "1 taza de arroz blanco"],
        "recipe": ["Mise en place: pesa todo.", "El Toque de Fuego: cocina 8 min.", "Montaje: sirve."]}]}]
    assert go._apply_budget_cheapen_pass(dias, {"budget": "low"}, force=True) >= 2
    m = dias[0]["meals"][0]
    assert m["name"] == "Arroz frío con filete de pescado blanco estilo ceviche", m["name"]
    assert "1 cda de Maní picado" in m["ingredients"], m["ingredients"]


def test_ganchos_y_aire_en_el_god_file():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert src.count('__import__("presupuesto_texto").sustituciones_unicas(result.get("days"))') == 2
    assert src.count('__import__("presupuesto_texto").sustituir(') >= 4
    assert "for s in (_bm.get(\"_budget_substitutions\") or [])" not in src
    assert "tooltip-anchor: P1-PLAN-LOTE-287-PRESUPUESTO-TEXTO" in (_BACKEND / "presupuesto_texto.py").read_text(encoding="utf-8")
