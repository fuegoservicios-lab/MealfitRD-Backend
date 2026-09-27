# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-547 · 2026-09-27] La cafeína y el tabaco que manda el asistente llegan al prompt.

Auditoría del formulario: `build_habits_prompt` comparaba la cafeína con «diario» y el tabaco con «semanal/diario», pero
el asistente (QHabits.jsx) manda «ninguna / 1-2 tazas/día / 3-4 tazas/día / 5+ tazas/día» y «no / ocasional / diario».
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from condition_rules import build_habits_prompt  # noqa: E402

_FRONT = _BACKEND.parent / "frontend" / "src" / "components" / "assessment" / "questions" / "QHabits.jsx"


def test_las_tazas_del_asistente_activan_la_directiva():
    for v in ("1-2 tazas/día", "3-4 tazas/día", "5+ tazas/día"):
        assert "CAFEÍNA DIARIA DECLARADA" in build_habits_prompt({"habitCaffeine": v}), v
    assert "CAFEÍNA ALTA" in build_habits_prompt({"habitCaffeine": "5+ tazas/día"})
    assert "CAFEÍNA ALTA" not in build_habits_prompt({"habitCaffeine": "1-2 tazas/día"})
    assert build_habits_prompt({"habitCaffeine": "ninguna"}) == ""


def test_fumar_a_veces_tambien_cuenta():
    assert "TABACO DECLARADO" in build_habits_prompt({"habitSmoking": "ocasional"})
    assert build_habits_prompt({"habitSmoking": "no"}) == ""


def test_los_valores_del_asistente_son_los_que_se_leen():
    if not _FRONT.exists():
        return
    src = _FRONT.read_text(encoding="utf-8")
    for v in ("'1-2 tazas/día'", "'3-4 tazas/día'", "'5+ tazas/día'", "'ocasional'"):
        assert v in src, v
