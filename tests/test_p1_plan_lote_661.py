# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-661 · 2026-09-28] Proteínas asignadas sin usar: aviso si el día cumple su proteína.

Decisión del dueño (28-sep). Producción 14-27 sep: 6 de 36 rechazos eran «Día N omitió múltiples proteínas clave
asignadas» y regeneraban el plan entero aunque el día llegara a su proteína.
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import fidelidad_proteina as fp  # noqa: E402


def _plan(p1, p2):
    return {"macros": {"protein": "100g"},
            "days": [{"day": 1, "meals": [{"protein": p1 / 2}, {"protein": p1 / 2}]},
                     {"day": 2, "meals": [{"protein": p2 / 2}, {"protein": p2 / 2}]}]}


_E1 = "Día 1 omitió múltiples proteínas clave asignadas: ['tilapia', 'queso blanco fresco']"
_E2 = "Día 2 omitió múltiples proteínas clave asignadas: ['pescado', 'huevos enteros']"


def test_el_dia_que_cumple_su_proteina_pasa_a_aviso():
    plan = _plan(98, 60)
    quedan = fp.filtrar(plan, [_E1, _E2], {"mainGoal": "lose_fat"})
    assert quedan == [_E2]                                   # el día 2 se queda corto: sigue rechazando
    assert plan["_skeleton_fidelity_advisory"] == [_E1]


def test_sin_objetivo_de_proteina_no_se_puede_medir():
    plan = _plan(98, 98)
    plan["macros"] = {}
    assert fp.filtrar(plan, [_E1, _E2], {}) == [_E1, _E2]


def test_knob_apagado(monkeypatch):
    monkeypatch.setenv("MEALFIT_FIDELITY_ADVISORY_IF_PROTEIN_OK", "false")
    assert fp.filtrar(_plan(98, 98), [_E1], {}) == [_E1]


def test_el_revisor_filtra_antes_de_rechazar():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    i = src.index('skeleton_fidelity_errors = plan.get("_skeleton_fidelity_errors", [])')
    j = src.index('skeleton_fidelity_errors = __import__("fidelidad_proteina").filtrar(plan, skeleton_fidelity_errors, '
                  'form_data)')
    assert 0 < j - i < 200                                  # la línea siguiente: antes de que nadie la lea
