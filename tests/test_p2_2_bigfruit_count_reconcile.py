"""[P2-2-BIGFRUIT-COUNT-RECONCILE · 2026-07-10] (recipe plausibility roadmap, item P2-2) `_bigfruit_bare_count_serving`
(P1-BIGFRUIT-GRAM-HINT) ya corrige la MAGNITUD del fantasma ("1 lechosa" 711kcal → "1 lechosa (200g)" 71kcal) —
pero el DISPLAY sigue diciendo "1 lechosa" (unidad ENTERA), y una lechosa real pesa 1-3 kg: el usuario lee
"compra 1 lechosa" cuando en realidad necesita ~⅙ de una. Fix ADITIVO display-only (NO toca
`ingredients_raw` ni el formato exacto ya testeado por `test_p1_bigfruit_gram_hint.py` — solo AÑADE un
descriptor fraccionario cuando el peso de la fruta ENTERA es resoluble).
"""
from __future__ import annotations

import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

with open(os.path.join(_BACKEND, "graph_orchestrator.py"), encoding="utf-8") as f:
    _GO = f.read()


def test_marker_present():
    assert "P2-2-BIGFRUIT-COUNT-RECONCILE" in _GO


def test_function_defined():
    assert "def annotate_bigfruit_fractional_hint(plan_data" in _GO


class _StubDB:
    """El BARE count ('1 lechosa madura', sin paréntesis) resuelve a 1200g de fruta entera
    (200g de porción / 1200g entera ≈ 1/6) — mimetiza el catálogo real (P1-BIGFRUIT-GRAM-HINT:
    'el hint parenteral GANA sobre el conteo' — por eso esta función SIEMPRE consulta el string
    SIN el paréntesis para obtener el peso de la fruta ENTERA)."""
    def grams_from_ingredient_string(self, s):
        return 1200.0 if "lechosa" in str(s).lower() else None

    def lookup(self, s):
        return object() if "lechosa" in str(s).lower() else None


def _plan(ingredient_line):
    return {"days": [{"day": 1, "meals": [{"meal": "Merienda", "name": "X",
                                            "ingredients": [ingredient_line],
                                            "ingredients_raw": [ingredient_line]}]}]}


def test_annotates_fraction_for_bigfruit_serving_line():
    import graph_orchestrator as g
    pd = _plan(f"1 lechosa madura ({g.BIGFRUIT_SERVING_G}g)")
    n = g.annotate_bigfruit_fractional_hint(pd, db=_StubDB())
    assert n >= 1
    line = pd["days"][0]["meals"][0]["ingredients"][0]
    assert f"({g.BIGFRUIT_SERVING_G}g)" in line, "el hint parseable original NO se toca"
    assert "≈" in line or "aprox" in line.lower()


def test_raw_untouched_by_annotation():
    import graph_orchestrator as g
    pd = _plan(f"1 lechosa madura ({g.BIGFRUIT_SERVING_G}g)")
    g.annotate_bigfruit_fractional_hint(pd, db=_StubDB())
    assert pd["days"][0]["meals"][0]["ingredients_raw"][0] == f"1 lechosa madura ({g.BIGFRUIT_SERVING_G}g)"


def test_noop_on_non_bigfruit_line():
    import graph_orchestrator as g
    pd = _plan("200 g de pollo")
    assert g.annotate_bigfruit_fractional_hint(pd, db=_StubDB()) == 0


def test_noop_when_whole_fruit_weight_unresolvable():
    import graph_orchestrator as g

    class _NoResolveDB:
        def grams_from_ingredient_string(self, s):
            return None

        def lookup(self, s):
            return None
    pd = _plan(f"1 lechosa madura ({g.BIGFRUIT_SERVING_G}g)")
    assert g.annotate_bigfruit_fractional_hint(pd, db=_NoResolveDB()) == 0


def test_failsafe_on_malformed_input():
    import graph_orchestrator as g
    assert g.annotate_bigfruit_fractional_hint({}, db=_StubDB()) == 0
    assert g.annotate_bigfruit_fractional_hint(None, db=_StubDB()) == 0
