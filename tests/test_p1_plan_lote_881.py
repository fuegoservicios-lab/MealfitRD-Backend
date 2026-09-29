# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-881 · 2026-09-29] El paso del 806 concuerda su verbo: «Bate el huevo… hasta que cuaje».

Corpus del 863 (planes guardados que recuperan el paso): «Bate el huevo, viértelo en la sartén… cocínalo… hasta que
cuajen por completo».
"""
from __future__ import annotations


def test_el_paso_del_huevo_concuerda_su_verbo():
    """[P1-PLAN-LOTE-881] el paso del 806 decía «Bate el huevo… cocínalo… hasta que cuajen» (corpus del 863)."""
    import huevo_sin_coccion as hs
    uno = hs.paso({"ingredients": ["1 huevo"], "recipe": ["Mise en place: corta el tomate."]})
    assert "hasta que cuaje por completo" in uno and "cuajen" not in uno, uno
    varios = hs.paso({"ingredients": ["3 huevos"], "recipe": ["Mise en place: bate 3 huevos."]})
    assert "hasta que cuajen por completo" in varios, varios
