"""[P2-SUBST-UNIT-DEDUP · 2026-07-05] "½ filete de Filete de pescado blanco (74g)" — screenshot
del plan vivo 23c958bb. La sustitución de presupuesto (mero→"Filete de pescado blanco") reemplaza
el token dentro de una línea cuya UNIDAD líder ya es el sustantivo del candidato → duplicación.
`_dedup_unit_noun_collision` colapsa "<unidad> de <misma-unidad> de" en línea/raw/nombre/pasos.
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

with open(os.path.join(_BACKEND, "graph_orchestrator.py"), encoding="utf-8") as f:
    _GO = f.read()


def _dedup(s):
    import graph_orchestrator as go
    return go._dedup_unit_noun_collision(s)


def test_filete_collision_collapsed():
    assert _dedup("½ filete de Filete de pescado blanco (74g)") == "½ filete de pescado blanco (74g)"


def test_plural_unit_preserved():
    assert _dedup("2 lonjas de lonja de jamón") == "2 lonjas de jamón"


def test_case_and_accent_insensitive():
    assert _dedup("1 pieza de Pieza de yuca") == "1 pieza de yuca"


def test_clean_lines_untouched():
    for s in ("½ filete de pescado blanco (74g)", "1 taza de melón", "2 lonjas de queso blanco (40g)",
              "1 taza de tazón mixto"):
        assert _dedup(s) == s


def test_applied_in_driver_aware_substitution():
    i = _GO.index("[P1-BUDGET-DRIVER-AWARE] Sustitución driver-aware")
    win = _GO[max(0, i - 3500):i]
    assert win.count("_dedup_unit_noun_collision") >= 3, \
        "el colapso debe aplicarse a línea, raw y nombre en la sustitución driver-aware"
