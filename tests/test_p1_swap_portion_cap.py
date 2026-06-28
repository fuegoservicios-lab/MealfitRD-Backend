"""[P1-SWAP-PORTION-CAP · 2026-06-27] Cierra la última brecha de paridad S1↔S3: el swap de un plato individual
(swap_meal en agent.py) corría MUCHOS backstops (clinical/alérgeno, renal, food-safety, condition-subst,
appetibility) pero NO los caps de porción DETERMINISTAS (DM2 almidón alto-IG + bariátrica queso/yogurt/fruta/
aguacate + volumen del pouch) que S1 (assemble) y S2 (regenerate-day) sí aplican. El swap solo tenía slot-target +
prompt → el LLM podía colar 5 lonjas de queso en una cena bariátrica sin recorte determinista.

Fix: tras condition_substitution_backstop, antes de appetibility, swap_meal aplica cap_dm2_high_gi_portions +
cap_bariatric_portions sobre el plato; como el recorte de lácteo baja proteína, re-cierra el piso del slot con
proteína animal densa NO-láctea (espejo de FASE A; renal → skip). Test parser-based (swap_meal necesita LLM/DB
real; anclamos el contrato en el source de prod). tooltip-anchor: P1-SWAP-PORTION-CAP
"""
from __future__ import annotations

import re
from pathlib import Path

import agent as _agent_mod

_BACKEND = Path(_agent_mod.__file__).resolve().parent
_AGENT = (_BACKEND / "agent.py").read_text(encoding="utf-8")


def _swap_meal_body() -> str:
    i = _AGENT.index("def swap_meal(")
    nxt = re.search(r"\ndef ", _AGENT[i + 10:])
    return _AGENT[i: i + 10 + (nxt.start() if nxt else len(_AGENT))]


def test_marker_present():
    assert "P1-SWAP-PORTION-CAP" in _swap_meal_body()


def test_applies_both_portion_caps():
    body = _swap_meal_body()
    assert "cap_dm2_high_gi_portions" in body
    assert "cap_bariatric_portions" in body


def test_reclose_protein_post_cap_non_dairy_renal_skip():
    """Tras el cap (que recorta lácteo=proteína) re-cierra el piso con proteína NO-láctea, y renal hace skip."""
    body = _swap_meal_body()
    # re-cierre usa el closer + candidatos, excluye lácteos, y respeta renal
    assert "_close_protein_gap_for_meal" in body or "_close_pc" in body
    assert "not _renal_capped" in body
    assert "queso" in body and "yogur" in body  # exclusión de lácteos en los candidatos del re-cierre


def test_runs_after_condition_subst_before_appetibility():
    """Orden correcto en la cadena de backstops (paridad con el orden de S1). Nota: hay 2 bloques
    P1-UPDATE-APPETIBILITY en swap_meal (clash mid-función + namefix final); el cap va antes del FINAL (rindex)."""
    body = _swap_meal_body()
    i_cond = body.index("P2-UPDATE-CONDITION-SUBST")
    i_cap = body.index("P1-SWAP-PORTION-CAP")
    i_appet_final = body.rindex("P1-UPDATE-APPETIBILITY")
    assert i_cond < i_cap < i_appet_final


def test_fail_open():
    """El cap de porción NUNCA debe bloquear el swap (fail-open, igual que los otros backstops)."""
    body = _swap_meal_body()
    seg = body[body.index("P1-SWAP-PORTION-CAP"):body.rindex("P1-UPDATE-APPETIBILITY")]
    assert "no bloquea" in seg and "except Exception" in seg
