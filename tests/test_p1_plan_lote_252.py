# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-252 · 2026-09-25] El escáner reconoce los alimentos de las demás clases de alergia que faltaban.

Sonda del lote 250 extendida a cada chip: «15 g de granola» (fila del catálogo; avena) pasaba limpia para un celíaco,
«1 cda de salsa de soya» (trigo) también, «Natilla» (yema) para un alérgico al huevo, «tamari» (soya), «tahín» (sésamo),
«macadamia» (frutos secos) y «lactosuero» (lácteos).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import graph_orchestrator as go  # noqa: E402
import vocabulario_alergenos as va  # noqa: E402

_CHIP = {"gluten": "Gluten", "huevo": "Huevo", "lacteos": "Lacteos", "frutos secos": "Frutos Secos", "soya": "Soya",
         "sesamo": "Sesamo"}


def _plan(*ings):
    return {"days": [{"day": 1, "meals": [{"meal": "Almuerzo", "name": "Plato", "ingredients": list(ings)}]}]}


@pytest.mark.parametrize("clase,termino", [(c, t) for c, ts in va.EXTRA.items() for t in ts])
def test_el_chip_ve_el_alimento(clase, termino):
    assert go._scan_allergen_violations(_plan(f"1 porción de {termino}"), [_CHIP[clase]]), (clase, termino)


@pytest.mark.parametrize("termino", va.EXTRA["huevo"] + va.EXTRA["lacteos"])
def test_la_dieta_vegana_tambien(termino):
    assert go._scan_diet_violations(_plan(f"1 porción de {termino}"), "vegano"), termino


@pytest.mark.parametrize("ing", ["15 g de granola", "1 cda de salsa de soya baja en sodio", "1 wrap integral",
                                 "2 pitas", "1 cda de tamari"])
def test_formas_reales(ing):
    assert go._scan_allergen_violations(_plan(ing), ["Gluten", "Soya"]), ing


@pytest.mark.parametrize("ing", ["1 cda de salsa de soya sin gluten", "15 g de granola sin gluten"])
def test_sin_gluten_sigue_excusado(ing):
    assert not go._scan_allergen_violations(_plan(ing), ["Gluten"]), ing


@pytest.mark.parametrize("ing", ["100 g de harina de maiz precocida", "1 arepa", "2 tortillas de maiz", "1 casabe",
                                 "1 cucharada de aceite de oliva", "100 g de pitahaya", "1 cda de tamarindo"])
def test_sin_falsos_positivos(ing):
    assert not go._scan_allergen_violations(_plan(ing), ["Gluten", "Soya", "Huevo", "Sesamo", "Frutos Secos"]), ing


def test_la_salsa_de_soya_es_gluten_escondido():
    assert go._scan_allergen_violations(_plan("1 cda de salsa de soya baja en sodio"), ["Gluten"])
    assert go._scan_allergen_violations(_plan("1 cda de salsa de soja"), ["Gluten"])
    assert not go._scan_allergen_violations(_plan("1 cda de salsa de soya sin gluten"), ["Gluten"])


def _clases(decl):
    from constants import strip_accents
    exp = go._expand_allergy_declarations([decl])
    return {c for c, syns in go._ALLERGEN_SYNONYMS.items()
            if syns and {strip_accents(str(s)).lower() for s in syns} <= exp}


@pytest.mark.parametrize("decl,esperado", [
    # La salsa de soya se BUSCA con gluten declarado, pero declarar «Soya» no declara gluten (pre-fix: sin pan ni avena).
    ("Soya", {"soya"}), ("soja", {"soya"}), ("Gluten", {"gluten"}),
    # Compuestos que no entraron a propósito: la palabra suelta se declara para otra cosa.
    ("papas", set()), ("buey", set()), ("callos", set()), ("tortilla", {"gluten"}), ("conchas", {"gluten"}),
    ("Mariscos", {"mariscos"}), ("Pescado", {"pescado"}), ("Huevo", {"huevo", "huevos"}),
    ("flan", {"lacteos", "lactosa", "huevo", "huevos"}), ("teriyaki", {"soya", "gluten"}),
])
def test_una_declaracion_no_arrastra_otra_clase(decl, esperado):
    assert _clases(decl) == esperado, (decl, _clases(decl))


def test_el_filtro_de_catalogo_saca_granola_y_natilla():
    from constants import _get_fast_filtered_catalogs
    for chip, fila in (("Gluten", "granola"), ("Huevo", "natilla")):
        pools = _get_fast_filtered_catalogs((chip,), (), "balanced")
        vivos = [str(x).lower() for pool in pools for x in pool]
        assert not any(fila in v for v in vivos), (chip, fila)


def test_ancla():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert src.count('__import__("vocabulario_alergenos")') == 11
    assert (_BACKEND / "constants.py").read_text(encoding="utf-8").count('__import__("vocabulario_alergenos")') == 6
    assert "P1-PLAN-LOTE-252-VOCABULARIO-ALERGENOS" in (_BACKEND / "vocabulario_alergenos.py").read_text(encoding="utf-8")


def test_marker():
    import re
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"',
                  (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 252
