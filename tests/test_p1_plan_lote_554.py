# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-554 · 2026-09-27] «Actualizar platos» ve la dieta y los rechazos, y guarda tras la última palabra.

Auditoría del formulario: el `_clin_form` con que `/regenerate-day` llama al cerrador de FASE A
(`_repair_protein_floor_post_caps`) no llevaba `dietType`, `dislikes`, `country` ni `mainGoal`, y el endpoint no tenía
ningún chequeo clínico después: vegetariana + Nevera con pollo ⇒ «pechuga de pollo» guardada.
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


def test_fase_a_recibe_dieta_rechazos_pais_y_objetivo():
    b = _cuerpo("api_regenerate_day")
    i = b.find("_clin_form = {")
    bloque = b[i:b.find("_day_wrap = [", i)]
    for k in ('"dietType"', '"dislikes"', '"otherDislikes"', '"country"', '"mainGoal"'):
        assert k in bloque, k
    assert "profile_with_free_text(_clin_form)" in bloque


def test_la_ultima_palabra_antes_de_guardar_el_dia():
    b = _cuerpo("api_regenerate_day")
    i_ctx = b.find("_rp_ctx554 = _bcfp554(verified_user_id)")
    i_mut = b.find("def _day_mutator(pd: dict) -> dict:")
    i_rp = b.find('__import__("restricciones_finales").retirar_prohibidos(')
    i_listas = b.find('pd, verified_user_id, surface="regen_day", plan_id_hint=plan_id')
    assert -1 not in (i_ctx, i_mut, i_rp, i_listas), (i_ctx, i_mut, i_rp, i_listas)
    assert i_ctx < i_mut < i_rp < i_listas, "el perfil se lee fuera del lock; la pasada, antes de derivar las listas"


def test_el_cerrador_con_la_dieta_no_elige_carne_para_una_vegetariana():
    import graph_orchestrator as go
    from constants import alergias_y_rechazos
    fd = go.profile_with_free_text({"dietType": "vegetarian", "allergies": [], "dislikes": [], "otherDislikes": "atún"})
    assert "atún" in [d.lower() for d in alergias_y_rechazos(fd)]
