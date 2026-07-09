"""[P1-SELF-CRITIQUE-MASTER-KNOB · 2026-07-09] Kill-switch del nodo self_critique entero, para A/B
y para la eventual restructuración (Fase 2). Default True = comportamiento actual intacto. NO se
cambia el default sin canario: self_critique aún posee detectores deterministas que el gate del
reviewer no tiene (staples cross-día, slot-incoherence), a portar antes de apagarlo.

Parser-based: el early-return por el knob debe ir al TOPE del nodo (antes del attempt-guard) para
que apague el nodo por completo.
"""
import os

_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(*parts):
    with open(os.path.join(*parts), encoding="utf-8") as f:
        return f.read()


def _src():
    return _read(_BACKEND, "graph_orchestrator.py")


def test_knob_defined():
    assert 'MEALFIT_SELF_CRITIQUE_ENABLED' in _src(), "falta el knob MEALFIT_SELF_CRITIQUE_ENABLED"


def test_marker_present():
    assert 'SELF-CRITIQUE-MASTER-KNOB' in _src(), "falta el tooltip-anchor SELF-CRITIQUE-MASTER-KNOB"


def test_early_return_at_top_of_node():
    """El `if not SELF_CRITIQUE_ENABLED: return {}` debe ir dentro de self_critique_node y ANTES
    del guard de attempt (para apagar el nodo entero, no solo el intento 1)."""
    src = _src()
    node = src[src.find("async def self_critique_node"):]
    node = node[: node.find("\nasync def ", 1) if node.find("\nasync def ", 1) > 0 else 6000]
    pos_knob = node.find("if not SELF_CRITIQUE_ENABLED")
    pos_attempt = node.find('state.get("attempt", 1) > 1')
    assert pos_knob >= 0, "falta el early-return `if not SELF_CRITIQUE_ENABLED` en self_critique_node"
    assert pos_attempt >= 0, "no se encontró el guard de attempt en self_critique_node"
    assert pos_knob < pos_attempt, "el kill-switch debe ir ANTES del guard de attempt"
