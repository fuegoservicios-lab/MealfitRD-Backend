# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-212 · 2026-09-24] V7a: el paso que pide MENOS piezas de las que compra la lista se alinea, con plural.

Decisión del dueño del 24-sep («soluciónalo»), que reemplaza la del 14-sep («el paso pide MENOS» no se corrige). Ver
`v7a_plural.py`.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import culinary_coherence as cc  # noqa: E402
import recipe_contract as rc  # noqa: E402
import v7a_plural as vp  # noqa: E402


def _row(name, aliases=(), category="vegetal"):
    return {"name": name, "aliases": list(aliases), "category": category, "ready_to_eat": False,
            "prep_methods": ["hervir", "saltear", "crudo"]}


_INDEX = cc.build_culinary_index([_row("Tomate", ["tomates"]), _row("Cebolla"), _row("Plátano verde"), _row("Huevo", ["huevos"]),
                                  _row("Tortilla integral"), _row("Pechuga de pollo", [], "proteina"), _row("Limón"),
                                  _row("Ají morrón")])


@pytest.mark.parametrize("sing,plu", [("tomate", "tomates"), ("cebolla", "cebollas"), ("limón", "limones"),
                                      ("cebollín", "cebollines"), ("ají", "ajíes"), ("nuez", "nueces"), ("pan", "panes"),
                                      ("pastel", "pasteles"), ("orden", None), ("lunes", None)])
def test_plural_del_espanol(sing, plu):
    assert vp.plural(sing) == plu


def test_plural_del_nombre_y_sus_calificativos():
    assert vp.plural_span("plátano verde") == "plátanos verdes"
    assert vp.plural_span("pechuga de pollo") == "pechugas de pollo"
    assert vp.plural_span("tortilla integral") == "tortillas integrales"
    assert vp.plural_span("ají morrón") is None, "un calificativo fuera del léxico no se adivina"


@pytest.mark.parametrize("lista,paso,esperado", [
    (["1½ tomates"], "Mise en place: corta ½ tomate en cubos y resérvalo.",
     "Mise en place: corta 1½ tomates en cubos y resérvalos."),
    (["2 cebollas"], "Mise en place: pica ½ de cebolla.", "Mise en place: pica 2 cebollas."),
    (["2 plátanos verdes"], "Mise en place: pela el ½ plátano verde.", "Mise en place: pela los 2 plátanos verdes."),
    (["3 huevos"], "Mise en place: bate 1 huevo.", "Mise en place: bate 3 huevos."),
])
def test_el_paso_que_pide_menos_se_alinea_con_plural(lista, paso, esperado):
    m = {"ingredients": lista, "recipe": [paso]}
    r = rc.reconcile_step_quantities(m, _INDEX)
    assert m["recipe"][0] == esperado and r["concordancia"] == 1 and r["sin_reparar"] == {}
    assert rc.reconcile_step_quantities(m, _INDEX)["reescritas"] == 0, "idempotente"


def test_lo_que_no_sabe_pluralizar_sigue_declarado():
    m = {"ingredients": ["2 ajíes morrones"], "recipe": ["Mise en place: corta 1 ají morrón."]}
    r = rc.reconcile_step_quantities(m, _INDEX)
    assert m["recipe"][0] == "Mise en place: corta 1 ají morrón." and r["sin_reparar"] == {"gramatical": 1}


def test_un_reparto_entre_pasos_sigue_sin_adivinarse():
    m = {"ingredients": ["3 tomates"], "recipe": ["Mise en place: pica 1 tomate.", "Montaje: añade 1 tomate en rodajas."]}
    r = rc.reconcile_step_quantities(m, _INDEX)
    assert r["sin_reparar"] == {"reparto": 2} and m["recipe"][0] == "Mise en place: pica 1 tomate."


def test_contra_el_peso_aproximado_con_una_mencion_tambien_se_sube():
    m = {"ingredients": ["½ pechuga de pollo (≈100 g)"], "recipe": ["Mise en place: corta 50 g de pechuga de pollo en cubos."]}
    rc.reconcile_step_quantities(m, _INDEX)
    assert m["recipe"][0] == "Mise en place: corta 100 g de pechuga de pollo en cubos."


def test_knob_apagado_vuelve_la_decision_del_14_de_septiembre(monkeypatch):
    monkeypatch.setenv("MEALFIT_CONTRACT_V7A", "false")
    m = {"ingredients": ["3 tomates"], "recipe": ["Mise en place: pica 1 tomate."]}
    r = rc.reconcile_step_quantities(m, _INDEX)
    assert m["recipe"][0] == "Mise en place: pica 1 tomate." and r["sin_reparar"] == {"gramatical": 1}


def test_la_decision_quedo_escrita():
    doc = (_BACKEND / "docs" / "decisiones_dueno_2026_09_14.md").read_text(encoding="utf-8")
    assert "P1-PLAN-LOTE-212" in doc


def test_marker():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"',
                  (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 212 and m.group(2) >= "2026-09-24"
