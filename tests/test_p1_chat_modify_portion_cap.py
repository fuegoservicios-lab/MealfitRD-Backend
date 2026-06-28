"""[P1-CHAT-MODIFY-PORTION-CAP · 2026-06-27] 4ta y última superficie de la paridad clínica: chat-modify
(`tools.execute_modify_single_meal`) tiene PATH PROPIO (no reusa swap_meal) y espejaba sus backstops (clinical/
alérgeno, renal, food-safety, condition-subst, appetibility) PERO igual le faltaban los caps de porción
deterministas (cap_dm2 + cap_bariatric) → el LLM podía colar 5 lonjas de queso al modificar un plato por chat.

Fix: tras condition_substitution_backstop, antes de appetibility, aplica ambos caps + re-cierre del piso del slot
con proteína animal densa NO-láctea (espejo FASE A, max_add_g=90; renal → skip). Mismo tooltip-anchor lógico
(P1-SWAP-PORTION-CAP) que el fix de swap. Test parser-based. tooltip-anchor: P1-CHAT-MODIFY-PORTION-CAP
"""
from __future__ import annotations

import re
from pathlib import Path

import tools as _tools_mod

_BACKEND = Path(_tools_mod.__file__).resolve().parent
_TOOLS = (_BACKEND / "tools.py").read_text(encoding="utf-8")


def _modify_body() -> str:
    i = _TOOLS.index("def execute_modify_single_meal")
    nxt = re.search(r"\ndef ", _TOOLS[i + 10:])
    return _TOOLS[i: i + 10 + (nxt.start() if nxt else len(_TOOLS))]


def test_applies_both_portion_caps():
    body = _modify_body()
    assert "cap_dm2_high_gi_portions" in body
    assert "cap_bariatric_portions" in body


def test_logs_modify_specific():
    """El log debe identificar la superficie chat-modify (no swap), para no confundir telemetría."""
    body = _modify_body()
    assert "plato de modify recortado" in body


def test_reclose_protein_non_dairy_renal_skip():
    body = _modify_body()
    assert "_close_protein_gap_for_meal" in body or "_close_pc_m" in body
    assert "renal_protein_cap" in body  # el skip renal usa el flag del plan
    assert "queso" in body and "yogur" in body  # exclusión de lácteos en el re-cierre


def test_runs_after_condition_subst_before_appetibility():
    body = _modify_body()
    i_cond = body.index("P2-UPDATE-CONDITION-SUBST")
    i_cap = body.index("P1-SWAP-PORTION-CAP")
    i_appet = body.rindex("P1-UPDATE-APPETIBILITY")
    assert i_cond < i_cap < i_appet


def test_fail_open():
    body = _modify_body()
    seg = body[body.index("P1-SWAP-PORTION-CAP"):body.rindex("P1-UPDATE-APPETIBILITY")]
    assert "no bloquea" in seg and "except Exception" in seg


def test_does_not_reuse_swap_meal():
    """Documenta el por qué del fix duplicado: chat-modify NO llama swap_meal (path propio), así que el fix de
    swap NO se hereda — hay que portarlo explícitamente."""
    body = _modify_body()
    assert "swap_meal(" not in body
