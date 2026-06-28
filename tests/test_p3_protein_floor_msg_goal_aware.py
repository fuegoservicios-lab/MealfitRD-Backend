"""[P3-PROTEIN-FLOOR-MSG-GOAL-AWARE · 2026-06-28] El usuario (objetivo MANTENIMIENTO + bariátrica) veía el rechazo
"DÉFICIT DE PROTEÍNA (rechazo clínico para ganancia muscular)" — el motivo estaba HARDCODEADO a 'ganancia muscular'
sin importar el objetivo real, y el % mostrado era 90% aunque para bariátrica el gate real es 80%. Ambos textos
engañaban el diagnóstico (el piso aplica a TODOS los objetivos; protege contra sub-entrega). Fix: motivo derivado del
objetivo real + % efectivo (bariátrico-aware).
"""
from __future__ import annotations

from pathlib import Path

import graph_orchestrator as g

_BACKEND = Path(g.__file__).resolve().parent
_SRC = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")


def _review_body():
    # bloque del gate del piso en review_plan_node (donde se arma el mensaje)
    i = _SRC.index("_short_days = _protein_floor_shortfall(plan, renal_capped=_renal_capped_plan")
    return _SRC[i:i + 2200]


def test_message_not_hardcoded_gain_muscle():
    body = _review_body()
    # el motivo ya NO está hardcodeado a 'ganancia muscular'
    assert "rechazo clínico para ganancia muscular" not in body
    assert "_motivo" in body and "adecuación proteica diaria" in body


def test_message_goal_aware_branch():
    body = _review_body()
    # gain_muscle → "ganancia muscular"; otro objetivo → genérico
    assert 'gain_muscle' in body and 'ganancia muscular' in body  # rama gain_muscle
    assert "(form_data or {}).get(\"mainGoal\")" in body or "form_data or {}).get('mainGoal'" in body


def test_pct_is_effective_not_hardcoded_90():
    body = _review_body()
    # el % mostrado usa el efectivo (bariátrico-aware), no PROTEIN_FLOOR_HARD_PCT fijo
    assert "_eff_pct" in body
    assert "PROTEIN_FLOOR_HARD_PCT_BARIATRIC" in body
    assert "int(_eff_pct*100)" in body or "int(_eff_pct * 100)" in body
