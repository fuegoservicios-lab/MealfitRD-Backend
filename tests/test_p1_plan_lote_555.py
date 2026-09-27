# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-555 · 2026-09-27] El coach que cambia un plato ve la dieta y los rechazos.

Auditoría del formulario: en `execute_modify_single_meal` los dos cerradores de proteína (tope de porción y cierre por
comida) elegían con `_safe_high_density_proteins(_clin_allergies, …)` —sin dieta ni «no me gusta»— DESPUÉS del único
chequeo clínico, y el plato se guardaba sin volver a mirarlo.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

_TOOLS = (_BACKEND / "tools.py").read_text(encoding="utf-8")


def _cuerpo(fn):
    m = re.search(rf"\ndef\s+{re.escape(fn)}\s*\(", _TOOLS)
    assert m
    nxt = re.search(r"\n(?:@tool|def\s)", _TOOLS[m.start() + 1:])
    return _TOOLS[m.start():m.start() + 1 + nxt.start()] if nxt else _TOOLS[m.start():]


def test_los_cerradores_del_coach_llevan_dieta_y_rechazos():
    b = _cuerpo("execute_modify_single_meal")
    assert "_safe_pc_m(_clin_allergies" not in b and "_shdp_m(_clin_allergies" not in b
    assert b.count("diet=_clin_diet") >= 4, "pool y cierre, en los dos cerradores"
    assert b.count("allergies=_restr555()") >= 2


def test_la_ultima_palabra_antes_de_las_listas():
    b = _cuerpo("execute_modify_single_meal")
    i_cierre = b.find("_cpg_m(new_meal_data, float(_anchor_p)")
    i_rp = b.find('__import__("restricciones_finales").retirar_prohibidos(')
    i_listas = b.find("aggr_list")
    i_lock = b.find("update_plan_data_atomic(")
    assert -1 not in (i_cierre, i_rp, i_listas, i_lock), (i_cierre, i_rp, i_listas, i_lock)
    assert i_cierre < i_rp < i_listas < i_lock


def test_el_cierre_de_micros_ve_los_rechazos_tecleados():
    b = _cuerpo("execute_modify_single_meal")
    assert '"dislikes": _pwft_cm(_hp).get("dislikes") or []' in b
