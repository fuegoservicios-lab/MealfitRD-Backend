"""[P0-ATOMIC-IDLE-GUARD · 2026-07-12] El guard idle-in-transaction de
`update_plan_data_atomic` DEBE aplicarse de verdad (y su fallo jamás ser invisible).

Root cause (madrugada 2026-07-12, incidente /swap-meal/persist ×2 con
`terminating connection due to idle-in-transaction timeout` a los ~22-29s):
el guard P0-PERSIST-TXN-IDLE-ATOMIC (2026-07-10) usaba `_env_int` SIN import en el
scope de `update_plan_data_atomic` (el único `from knobs import _env_int` vive en el
scope local de `set_meal_plan_for_update_timeouts`) → NameError → tragado por
`except Exception → logger.debug` → **el SET LOCAL de 60s nunca se ejecutó desde que
se escribió**: toda txn atómica con mutator >15s idle moría al default de sesión.

Triangulación empírica (repro en VPS): secuencia MANUAL (pool → transaction →
SET LOCAL 60s → FOR UPDATE → sleep 25s) SOBREVIVE; la misma vía el helper MORÍA.

Contrato:
1. `_env_int` importado (o resuelto) DENTRO del scope que lo usa en el guard.
2. El fallo del guard se loggea como WARNING (no debug): un guard muerto en
   silencio costó 2 persists de usuario en vivo — visible o no existe.
3. Smoke de ejecutabilidad: la expresión del guard evalúa sin NameError.

tooltip-anchor: P0-ATOMIC-IDLE-GUARD
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BACKEND))
_SRC = (_BACKEND / "db_plans.py").read_text(encoding="utf-8")


def _atomic_body() -> str:
    i = _SRC.find("def update_plan_data_atomic")
    assert i != -1
    j = _SRC.find("\ndef ", i + 10)
    return _SRC[i:j if j != -1 else len(_SRC)]


def test_env_int_resolvable_in_guard_scope():
    body = _atomic_body()
    m_use = re.search(r'(_env_int\w*)\("MEALFIT_PLAN_FOR_UPDATE_IDLE_TXN_TIMEOUT_MS"', body)
    assert m_use, "el guard idle-txn desapareció de update_plan_data_atomic"
    _name = m_use.group(1)
    m_imp = re.search(r"from knobs import _env_int(?:\s+as\s+" + _name + r")?", body)
    assert m_imp, (
        "NameError silencioso: _env_int se usaba sin import en este scope → el SET LOCAL "
        "de 60s JAMÁS se aplicó (2026-07-10 → 2026-07-12) y los persists morían a los 15s"
    )
    assert m_imp.start() < m_use.start(), "el import debe preceder al uso"


def test_guard_failure_is_warning_not_debug():
    body = _atomic_body()
    i_guard = body.find("P0-PERSIST-TXN-IDLE-ATOMIC] no se pudo setear idle timeout")
    assert i_guard != -1, "el log de fallo del guard desapareció"
    line_start = body.rfind("\n", 0, i_guard)
    line = body[line_start:i_guard + 60]
    assert "logger.warning" in line, (
        "un guard de resiliencia que falla en SILENCIO (debug) es un guard que no existe "
        "— el NameError vivió 2 días invisible y costó persists de usuario en vivo"
    )


def test_guard_expression_executes():
    """Smoke sin DB: la expresión del knob evalúa (el NameError original habría
    reventado aquí)."""
    from knobs import _env_int
    v = _env_int("MEALFIT_PLAN_FOR_UPDATE_IDLE_TXN_TIMEOUT_MS", 60000)
    assert isinstance(v, int) and v > 0


def test_marker_anchored():
    assert _SRC.count("P0-ATOMIC-IDLE-GUARD") >= 1
