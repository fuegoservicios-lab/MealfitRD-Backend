"""[P3-1-BRAND-LOWERCASE-STRIP · 2026-07-10] (recipe plausibility roadmap, item P3-1) `_BRAND_PAREN_RE`
(dentro de `_polish_finalize_display`, regla 3: "marca del súper fuera de la receta") solo matchea
paréntesis cuyo PRIMER caracter es MAYÚSCULA (`[A-ZÁÉÍÓÚÑ]...`) — "(Campos)" se strippea correctamente,
pero "(jif)"/"(borges)" (evidencia visual, plan 564d6e4e: "2 tortillas de trigo", "1.25 cdas de
mantequilla de maní (jif)"; "1 cdta de aceite de oliva (borges)") quedan intactos porque la marca se
insertó en minúscula. Fix: whitelist curada de marcas conocidas en minúscula, además del patrón
case-sensitive existente (que sigue intacto para marcas ya bien capitalizadas).
"""
from __future__ import annotations

import graph_orchestrator as g


def test_marker_present():
    import os
    _here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(os.path.dirname(_here), "graph_orchestrator.py"), encoding="utf-8") as f:
        assert "P3-1-BRAND-LOWERCASE-STRIP" in f.read()


def test_strips_known_lowercase_brand_jif():
    d = [{"day": 1, "meals": [{"meal": "M", "name": "N",
                               "ingredients": ["1.25 cdas de mantequilla de maní (jif)"],
                               "ingredients_raw": ["1.25 cdas de mantequilla de maní (jif)"]}]}]
    g._polish_finalize_display(d)
    line = d[0]["meals"][0]["ingredients"][0]
    assert "(jif)" not in line.lower()
    assert "mantequilla" in line.lower()


def test_strips_known_lowercase_brand_borges():
    d = [{"day": 1, "meals": [{"meal": "M", "name": "N",
                               "ingredients": ["1 cdta de aceite de oliva (borges)"],
                               "ingredients_raw": ["1 cdta de aceite de oliva (borges)"]}]}]
    g._polish_finalize_display(d)
    line = d[0]["meals"][0]["ingredients"][0]
    assert "(borges)" not in line.lower()


def test_existing_capitalized_brand_still_stripped():
    """Regresión: el patrón case-sensitive existente sigue funcionando para marcas ya capitalizadas."""
    d = [{"day": 1, "meals": [{"meal": "M", "name": "N",
                               "ingredients": ["200 g de arroz blanco (Campos)"],
                               "ingredients_raw": ["200 g de arroz blanco (Campos)"]}]}]
    g._polish_finalize_display(d)
    assert "(Campos)" not in d[0]["meals"][0]["ingredients"][0]


def test_en_total_annotation_not_affected():
    """Regresión crítica: '(en total)' (P1-1-QTYSYNC-STALE-EN-TOTAL) NO debe ser tratado como marca."""
    d = [{"day": 1, "meals": [{"meal": "M", "name": "N",
                               "ingredients": ["1.25 cdas de mantequilla de maní (en total)"],
                               "ingredients_raw": ["1.25 cdas de mantequilla de maní (en total)"]}]}]
    g._polish_finalize_display(d)
    assert "(en total)" in d[0]["meals"][0]["ingredients"][0]
