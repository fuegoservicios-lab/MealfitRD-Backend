"""[P1-ADVERSARIAL-PAID-ONLY · 2026-07-09] El adversarial self-play (2 candidatos + juez LLM #5)
es una capa de calidad ADITIVA, no un gate de seguridad. Se restringe a tiers pagados: free/guest
caen al path single-candidate (su comportamiento normal hoy). Ahorra 2× day-gen + juez para el free
tier sin pérdida clínica. Knob MEALFIT_ADVERSARIAL_PAID_ONLY (default True). Rollback = False.

Parser-based: el gate DEBE ir ANTES del bloque `if use_adversarial:` (si va después, la doble
generación ya ocurrió). Ancla el contrato en el source.
"""
import os

_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(*parts):
    with open(os.path.join(*parts), encoding="utf-8") as f:
        return f.read()


def _src():
    return _read(_BACKEND, "graph_orchestrator.py")


def test_knob_defined():
    assert 'MEALFIT_ADVERSARIAL_PAID_ONLY' in _src(), "falta el knob MEALFIT_ADVERSARIAL_PAID_ONLY"


def test_marker_present():
    assert 'ADVERSARIAL-PAID-ONLY' in _src(), "falta el tooltip-anchor ADVERSARIAL-PAID-ONLY"


def test_gate_uses_tier():
    src = _src()
    # anclar en el GATE (no en la definición del knob, que también contiene el marker)
    seg = src[src.find("if use_adversarial and ADVERSARIAL_PAID_ONLY"):]
    assert seg, "no se encontró el gate `if use_adversarial and ADVERSARIAL_PAID_ONLY`"
    # el gate debe consultar el tier del usuario y compararlo contra PAID_TIERS
    assert "get_user_tier(" in seg[:500], "el gate debe resolver el tier con get_user_tier"
    assert "PAID_TIERS" in seg[:500], "el gate debe comparar contra PAID_TIERS"


def test_gate_runs_before_self_play_block():
    """El gate de tier debe ir ANTES del `if use_adversarial:` que dispara la self-play (2×
    day-gen). Si va después, la doble generación ya ocurrió y el gate es inútil."""
    src = _src()
    pos_gate = src.find("if use_adversarial and ADVERSARIAL_PAID_ONLY")
    pos_selfplay = src.find("\n    if use_adversarial:")
    assert pos_gate >= 0, "falta el gate `if use_adversarial and ADVERSARIAL_PAID_ONLY`"
    assert pos_selfplay >= 0, "no se encontró el bloque self-play `if use_adversarial:`"
    assert pos_gate < pos_selfplay, "el gate paid-only debe ir ANTES del bloque self-play"
