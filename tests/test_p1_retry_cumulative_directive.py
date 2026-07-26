"""[P1-RETRY-CUMULATIVE-DIRECTIVE · 2026-07-09] Directiva de retry acumulativa (anti whack-a-mole).

Forense plan vivo 1273aecb (angelobrito500, gain_muscle, 2026-07-09 00:10-00:18): 3 rechazos consecutivos,
cada uno por un gate DISTINTO —
  #1: cross-day-dish ('revoltillo' ×3) + "ninguna preparación transformada" + banda proteína 0.333
  #2: MISMA PROTEÍNA MISMO DÍA
  #3: MISMA PROTEÍNA MISMO DÍA otra vez
El `retry_reflection_node` construía la directiva SOLO con `rejection_reasons` del ÚLTIMO rechazo. Así el
intento #2 "arreglaba" el cross-day-dish pero introducía same-day-protein (que #1 no violaba → no estaba en
la directiva). Whack-a-mole: fija un gate, rompe otro, en bucle → 3-4 regeneraciones completas (~2 min c/u).

Fix: acumular las razones de TODOS los intentos previos (`_cumulative_rejection_reasons`, dedup por prefijo
normalizado) e inyectarlas como "RESTRICCIONES ACUMULADAS que DEBES respetar SIMULTÁNEAMENTE". Knob
`MEALFIT_RETRY_CUMULATIVE_DIRECTIVE` (default True).
"""
import os

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

with open(os.path.join(_BACKEND, "graph_orchestrator.py"), encoding="utf-8") as f:
    _GO = f.read()


# ───────────────────────── parser-based ─────────────────────────

def test_marker_present():
    assert "P1-RETRY-CUMULATIVE-DIRECTIVE" in _GO


def test_knob_default_on():
    assert (
        'RETRY_CUMULATIVE_DIRECTIVE_ENABLED = _env_bool("MEALFIT_RETRY_CUMULATIVE_DIRECTIVE", True)'
        in _GO
    )


def test_helper_and_state_key():
    assert "def _merge_rejection_reasons(" in _GO
    assert "_cumulative_rejection_reasons" in _GO


def test_directive_injects_accumulated_constraints():
    assert "RESTRICCIONES ACUMULADAS" in _GO
    # el nodo de retry debe persistir el set acumulado en el state
    i_node = _GO.index("async def retry_reflection_node")
    i_store = _GO.index('"_cumulative_rejection_reasons"', i_node)
    assert i_store - i_node < 4000, "retry_reflection_node debe escribir _cumulative_rejection_reasons"


# ───────────────────────── funcional ─────────────────────────

@pytest.fixture()
def g():
    import graph_orchestrator as _g
    return _g


def test_merge_dedups_same_reason(g):
    prior = ["MISMA PROTEÍNA REPETIDA EL MISMO DÍA (rechazo de variedad): ..."]
    current = ["MISMA PROTEÍNA REPETIDA EL MISMO DÍA (rechazo de variedad): ..."]
    out = g._merge_rejection_reasons(prior, current)
    assert len(out) == 1


def test_merge_accumulates_distinct_gates(g):
    """Caso vivo: #1 cross-day-dish+band, #2 same-day-protein → cumulative tiene AMBOS."""
    prior = [
        "MISMO PLATO REPETIDO ENTRE DÍAS (rechazo de variedad): 'revoltillo' en 3 días.",
        "PRECISIÓN DE MACROS BAJA: varios días fuera de banda, especialmente protein.",
    ]
    current = ["MISMA PROTEÍNA REPETIDA EL MISMO DÍA (rechazo de variedad): ..."]
    out = g._merge_rejection_reasons(prior, current)
    assert len(out) == 3
    # order-preserving: previas primero, luego la nueva
    assert out[0].startswith("MISMO PLATO")
    assert out[-1].startswith("MISMA PROTE")


def test_merge_key_is_whitespace_insensitive(g):
    prior = ["MISMA   PROTEÍNA  repetida  el mismo día blah blah blah blah blah blah"]
    current = ["misma proteína repetida el mismo día blah blah blah blah blah blah"]
    # mismo prefijo normalizado → dedup a 1
    assert len(g._merge_rejection_reasons(prior, current)) == 1


def test_merge_fail_safe(g):
    assert g._merge_rejection_reasons(None, None) == []
    assert g._merge_rejection_reasons([], ["x"]) == ["x"]
    assert g._merge_rejection_reasons(["a"], []) == ["a"]
    # dropea falsy
    assert g._merge_rejection_reasons(["", None], ["real"]) == ["real"]
