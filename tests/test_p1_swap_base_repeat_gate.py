"""[P1-SWAP-BASE-REPEAT-GATE · 2026-07-10] El swap con reason variety/similar/dislike no
puede devolver el MISMO plato-base con variación cosmética. + [P2-REGEN-DAY-PARTIAL-REBALANCE]
el rebalance del día escala hasta donde la Nevera dé (mitades) en vez de revertir todo.

Evidencia (screenshots + logs del owner, 2026-07-10): 3 regens seguidos del desayuno
devolvieron "Panqueques de Avena y Guineo …" (con Cottage → con Yogur → Maduro con Yogur) —
el prompt solo veta el nombre EXACTO y el LLM colapsa a la misma base con la despensa
disponible. Y el día regenerado quedaba band 0.5 (proteína/kcal 0.0) porque el rebalance
revertía TODO cuando el target completo excedía la Nevera → chips ámbar honestos pero
inarreglables sin comprar.

tooltip-anchor: P1-SWAP-BASE-REPEAT-GATE
"""
from __future__ import annotations

from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
_AGENT = (_BACKEND / "agent.py").read_text(encoding="utf-8")
_PLANS = (_BACKEND / "routers" / "plans.py").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Gate anti mismo-plato-base en el swap
# ---------------------------------------------------------------------------

def test_base_repeat_gate_wired_in_swap():
    i = _AGENT.find("P1-SWAP-BASE-REPEAT-GATE")
    assert i > 0, (
        "P1-SWAP-BASE-REPEAT-GATE: el swap volvió a permitir el mode collapse "
        "('Panqueques de Avena…' ×3 regens seguidos con variación cosmética)."
    )
    blk = _AGENT[i: i + 3200]
    assert "_head_dish_base_token" in blk, "SSOT del plato-base (mismo detector del gate cross-día)"
    assert "rejected_meal" in blk, "compara contra el plato ACTUAL que el usuario quiere cambiar"
    assert '"variety", "similar", "dislike"' in blk, (
        "solo aplica cuando el usuario pidió algo DISTINTO — time/cravings/weekend pueden conservar base"
    )
    assert "SWAP_SAME_BASE" in blk and "raise ValueError" in blk, (
        "el gate debe ser retryable (ValueError → tenacity) con directiva de cambiar la BASE"
    )
    assert 'os.environ.get("MEALFIT_SWAP_BASE_REPEAT_GATE", "true")' in blk, "knob default ON"


def test_base_repeat_gate_functional_same_base_detected():
    import sys
    sys.path.insert(0, str(_BACKEND))
    from graph_orchestrator import _head_dish_base_token
    from constants import strip_accents
    cur = _head_dish_base_token(strip_accents("Panqueques de Avena y Guineo con Queso Cottage y Arándanos".lower()))
    new = _head_dish_base_token(strip_accents("Panqueques de Avena y Guineo Maduro con Yogur y Arándanos".lower()))
    assert cur and new and cur == new, "el caso vivo del owner debe detectarse como mismo plato-base"
    distinto = _head_dish_base_token(strip_accents("Revoltillo de Huevos con Casabe".lower()))
    assert distinto != cur, "un cambio real de base (revoltillo) NO debe gatearse"


# ---------------------------------------------------------------------------
# 2. Rebalance parcial limitado por Nevera
# ---------------------------------------------------------------------------

def test_partial_rebalance_before_full_revert():
    i = _PLANS.find("P2-REGEN-DAY-PARTIAL-REBALANCE")
    assert i > 0, (
        "P2-REGEN-DAY-PARTIAL-REBALANCE: el rebalance del regen-day volvió al revert TOTAL "
        "— días band 0.5 con chips ámbar inarreglables cuando la Nevera no da para el target."
    )
    blk = _PLANS[i: i + 3200]
    assert "(0.5, 0.25)" in blk, "reintentos al 50% y 25% del delta antes del revert total"
    assert "_day_exceeds_pantry(_attempt_rb" in blk, (
        "cada intento parcial se re-valida contra la Nevera ORIGINAL (never-worse-than-current)"
    )
    assert "deepcopy(_pre_rb)" in blk, (
        "cada intento parte de una copia FRESCA del estado pre-rebalance (el rebalance muta in-place)"
    )
    assert "_pantry_limited = True" in _PLANS[i: i + 4200], (
        "el residual (total o parcial) sigue atribuyéndose a la Nevera para el aviso honesto"
    )
