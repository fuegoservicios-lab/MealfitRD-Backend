# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-553 · 2026-09-27] Guardar un cambio de plato respeta lo tecleado y pasa por la última palabra.

Auditoría del formulario: `/swap-meal/persist` hidrataba alergias y rechazos SÓLO de los chips; su finalizador vuelve a
añadir los alimentos que nombran los pasos DESPUÉS del re-chequeo — «Otra alergia: pimiento» + «sofríe con pimiento»
guardaba «100 g de pimiento» en el plato y en la lista de compras.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

_PLANS = (_BACKEND / "routers" / "plans.py").read_text(encoding="utf-8")


def _cuerpo(fn):
    m = re.search(rf"def\s+{re.escape(fn)}\s*\(", _PLANS)
    assert m
    nxt = re.search(r"\n(?:@router\.|@app\.|def\s)", _PLANS[m.start() + 1:])
    return _PLANS[m.start():m.start() + 1 + nxt.start()] if nxt else _PLANS[m.start():]


def test_el_persist_lee_alergias_y_rechazos_con_lo_tecleado():
    b = _cuerpo("api_swap_meal_persist")
    assert "_hp_ft553 = _pwft553(_hp_micro)" in b
    assert '"allergies": [str(a).strip() for a in (_hp_ft553.get("allergies") or [])' in b
    assert '"dislikes": _hp_ft553.get("dislikes") or []' in b
    assert '_persist_allergies = [str(a).strip() for a in (_hp_ft553.get("allergies") or [])' in b
    assert '_hp_micro.get("allergies")' not in b, "quedó una lectura de alergias sólo de chips"


def test_la_ultima_palabra_va_antes_de_derivar_las_listas():
    b = _cuerpo("api_swap_meal_persist")
    i_rp = b.find('__import__("restricciones_finales").retirar_prohibidos(')
    i_fin = b.find("finalize_single_meal_recipe_coherence as _fin_sp")
    i_listas = b.find('_rebuild_plan_shopping_lists_inline(\n                plan_data, verified_user_id, surface="swap_persist"')
    assert -1 not in (i_rp, i_fin, i_listas), (i_rp, i_fin, i_listas)
    assert i_fin < i_rp < i_listas


def test_lo_tecleado_llega_a_la_union():
    from graph_orchestrator import profile_with_free_text
    hp = {"allergies": [], "otherAllergies": "pimiento", "dislikes": ["Cebolla"], "otherDislikes": "cilantro"}
    ft = profile_with_free_text(hp)
    assert "pimiento" in [a.lower() for a in ft["allergies"]] and "cilantro" in [d.lower() for d in ft["dislikes"]]
    assert hp["allergies"] == []
