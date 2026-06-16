"""[P2-14 · 2026-06-16] Dedup behavior-preserving del reassign renal de proteína (gap-audit P2-14).

La lógica 'capear protein_str al cap renal + reasignar las kcal liberadas a grasa (DM2 comórbido) o
carbo iso-kcal' estaba TRIPLICADA: SITE A (`_cap_macros_dict_renal`, *_g+*_str, int(round)), SITE B
(`_apply_deterministic_clinical_layer`, inline) y SITE C (`assemble_plan_node`, inline). B y C eran
idénticos → extraídos a `_renal_reassign_macro` VERBATIM (round(), *_str + mirror). SITE A NO se toca.

Oracle de equivalencia: valores calculados a mano (la aritmética exacta de los inline B/C) + parser que
el helper se invoca exactamente 2 veces. La política carb-vs-fat NO cambia (decisión diferida a nutricionista).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_GO_SRC = (Path(__file__).resolve().parent.parent / "graph_orchestrator.py").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def go():
    import graph_orchestrator as _go
    return _go


# ───────────────────────── Oracle: aritmética exacta de los inline B/C ─────────────────────────
def test_diabetic_reassigns_freed_to_fat(go):
    am = {"protein_str": "150g", "fats_str": "60g", "carbs_str": "200g"}
    mir = {"protein": "150g", "fats": "60g", "carbs": "200g"}
    r = go._renal_reassign_macro(am, 80, 150.0, True, mir)
    # freed = (150-80)*4 = 280 kcal; fat += round(280/9)=31 → 60+31=91; carbo intacto
    assert r == "fat"
    assert am["protein_str"] == "80g" and am["fats_str"] == "91g" and am["carbs_str"] == "200g"
    assert mir["protein"] == "80g" and mir["fats"] == "91g" and mir["carbs"] == "200g"


def test_nondiabetic_reassigns_freed_to_carb(go):
    am = {"protein_str": "150g", "fats_str": "60g", "carbs_str": "200g"}
    r = go._renal_reassign_macro(am, 80, 150.0, False, None)  # mirror=None no debe lanzar
    # swap iso-kcal 1:1 g: carbo += (150-80)=70 → 200+70=270; grasa intacta
    assert r == "carb"
    assert am["protein_str"] == "80g" and am["carbs_str"] == "270g" and am["fats_str"] == "60g"


def test_mirror_none_safe(go):
    am = {"protein_str": "120g", "fats_str": "50g", "carbs_str": "180g"}
    assert go._renal_reassign_macro(am, 70, 120.0, True, None) == "fat"
    assert am["protein_str"] == "70g"  # (120-70)*4=200 kcal; fat += round(200/9)=22 → 72
    assert am["fats_str"] == "72g"


def test_uses_round_not_int_round(go):
    # SITE A usa int(round()); B/C (este helper) usa round() — verifica que NO truncó distinto.
    am = {"protein_str": "100g", "fats_str": "55g", "carbs_str": "150g"}
    go._renal_reassign_macro(am, 64, 100.0, True, None)
    # freed=(100-64)*4=144; 144/9=16.0 → 55+16=71
    assert am["fats_str"] == "71g"


# ───────────────────────── Parser: dedup real (1 def + 2 callsites, A intacto) ─────────────────────────
def test_helper_defined_and_anchored():
    assert "def _renal_reassign_macro(" in _GO_SRC
    assert "P2-14-RENAL-REASSIGN" in _GO_SRC


def test_helper_called_exactly_twice():
    # SITES B y C ahora delegan; SITE A (_cap_macros_dict_renal) NO usa el helper.
    calls = len(re.findall(r"_reassigned = _renal_reassign_macro\(", _GO_SRC))
    assert calls == 2, f"esperaba 2 callsites (B/C), encontré {calls}"
    # el inline viejo (_freed local + ramas) NO debe quedar en los call sites
    assert "_freed_kcal_renal = (_old_p_renal" not in _GO_SRC, "SITE C inline no eliminado"


def test_site_a_untouched(go):
    # SITE A sigue existiendo con su firma propia (int(round), *_g+*_str).
    assert "def _cap_macros_dict_renal(" in _GO_SRC
    assert "int(round(p))" in _GO_SRC
