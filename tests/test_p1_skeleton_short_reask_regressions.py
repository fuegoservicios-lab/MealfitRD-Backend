"""[P1-SKELETON-SHORT-REASK · 2026-09-02] Regresiones del propio fix, vistas en el primer run
con el binario nuevo (run f877831f, plan 19e8b509):

  1. La inserción de los helpers dejó `@_node_label("planner")` encima de un helper y el
     planificador salió como `node=None` en `llm_usage_events`. El decorador va pegado al nodo.
  2. El Día 1, re-corregido por el marker surgical DESPUÉS del estampado de fechas, se persistió
     sin `date` aunque el plan nunca pasó por la rama de relleno del guardrail. El estampado
     universal va al FINAL de `_apply_final_defense_guardrails` (único call site pre-persist).
"""
from __future__ import annotations

import re
from pathlib import Path

_SRC = (Path(__file__).resolve().parents[1] / "graph_orchestrator.py").read_text(encoding="utf-8")


def test_planner_node_keeps_its_label_right_above_the_def():
    m = re.search(r'@_node_label\("planner"\)\r?\n(async )?def plan_skeleton_node\(', _SRC)
    assert m, "el decorador del planner debe preceder INMEDIATAMENTE a plan_skeleton_node"
    bad = re.search(r'@_node_label\("planner"\)\s*\r?\n(#[^\n]*\r?\n)*def _skeleton_short_reask_directive', _SRC)
    assert not bad, "el decorador volvió a caer sobre el helper (node=None en usage events)"


def test_universal_stamp_runs_at_the_end_of_final_guardrails():
    i = _SRC.find("def _apply_final_defense_guardrails(")
    assert i != -1
    m = re.compile(r"\r?\n(async )?def ").search(_SRC, i + 10)
    body = _SRC[i:m.start()]
    k = body.rfind("_stamp_missing_day_dates(_fp_final, actual_form_data)")
    assert k != -1, "estampado universal pre-persist ausente"
    assert 'final_state.get("plan_result")' in body[k - 400:k]
    assert body.count("_stamp_missing_day_dates(") == 2, "rama de relleno + universal"
    # el universal es lo ÚLTIMO de la función (nada lo puede dejar atrás)
    tail = body[k:]
    assert "return" not in tail.split("except Exception as _stamp_final_err")[0]


def test_final_guardrails_has_a_single_pre_persist_call_site():
    calls = [m.start() for m in re.finditer(r"_apply_final_defense_guardrails\(", _SRC)]
    defs = [m.start() for m in re.finditer(r"def _apply_final_defense_guardrails\(", _SRC)]
    assert len(calls) - len(defs) == 1, "si aparece otro call site, el estampado universal debe seguir cubriéndolo"
