"""[P1-MACRO-SOLVER-DEFAULT-ON · 2026-06-26] El motor de precisión de macros
(el "cerebro dividido" determinista: solver LSQ de porciones + protein-closer +
techo de proteína + reconcite) debe estar ON POR DEFAULT EN CÓDIGO.

Contexto (auditoría 2026-06-26, gap #1): `MEALFIT_MACRO_SOLVER_ENABLED` tenía
default de código `False` y solo estaba ON por la env var del `.env` del VPS.
Modo de fallo: un redeploy con `.env` limpio (o cualquier entorno dev/CI sin la
var) revertía SILENCIOSAMENTE al porcionado "a ojo" del LLM — benchmark medido
proteína ~16% MAPE, solo ~24% de días con los 4 macros en banda — y dejaba muerto
TODO el cerebro dividido que cuelga del mismo gate (solver, cal-reconcile, closer,
techo de proteína). El canary documentado en `docs/macro_rollout_and_validation.md`
ya se corrió y validó (0% fallback, proteína ~7.3% MAPE, all-4-en-banda ↑), así que
el default de código se alineó con la config validada de prod (mismo patrón que
P1-MACRO-RECONCILE-DEFAULT para `MACRO_AWARE_RECONCILE`, 2026-06-18).

Este test es parser-based (no importa graph_orchestrator para no depender del env
de import) y ancla:
  1. El default del knob en el source es `True`.
  2. El tooltip-anchor `P1-MACRO-SOLVER-DEFAULT-ON` está presente (un rename del
     knob/comentario falla este test ANTES de cambiar producción).
  3. `.env.example` y la tabla de knobs del doc de rollout reflejan `True`
     (sin drift doc↔código).

Rollback intencional sin redeploy: `MEALFIT_MACRO_SOLVER_ENABLED=False` en el `.env`.
Si alguien revierte el DEFAULT DE CÓDIGO a False, debe actualizar esta razón aquí.
"""
from __future__ import annotations

import re
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_GO_SRC = (_BACKEND_ROOT / "graph_orchestrator.py").read_text(encoding="utf-8")


def test_macro_solver_default_is_true_in_code():
    """El knob maestro del motor de macros debe declararse con default `True`."""
    m = re.search(
        r'MACRO_SOLVER_ENABLED\s*=\s*_env_bool\(\s*["\']MEALFIT_MACRO_SOLVER_ENABLED["\']\s*,\s*(True|False)\s*\)',
        _GO_SRC,
    )
    assert m is not None, (
        "No se encontró la declaración de MACRO_SOLVER_ENABLED = "
        "_env_bool(\"MEALFIT_MACRO_SOLVER_ENABLED\", ...) en graph_orchestrator.py "
        "(¿renombrada? actualiza el ancla del test)."
    )
    assert m.group(1) == "True", (
        "MACRO_SOLVER_ENABLED tiene default de código False — el motor de precisión "
        "de macros se apagaría en cualquier entorno sin la env var (porcionado 'a ojo', "
        "proteína ~16% MAPE). Debe ser True (P1-MACRO-SOLVER-DEFAULT-ON). Para apagarlo "
        "en un entorno específico usa MEALFIT_MACRO_SOLVER_ENABLED=False en el .env, NO "
        "el default de código."
    )


def test_tooltip_anchor_present():
    """Un rename del knob/comentario debe romper este test antes que producción."""
    assert "P1-MACRO-SOLVER-DEFAULT-ON" in _GO_SRC, (
        "Falta el tooltip-anchor P1-MACRO-SOLVER-DEFAULT-ON en graph_orchestrator.py."
    )


def test_env_example_reflects_true():
    """`.env.example` no debe contradecir el default de código."""
    env_example = (_BACKEND_ROOT / ".env.example").read_text(encoding="utf-8")
    assert re.search(r"^\s*MEALFIT_MACRO_SOLVER_ENABLED\s*=\s*True\s*$", env_example, re.MULTILINE), (
        "`.env.example` debería documentar MEALFIT_MACRO_SOLVER_ENABLED=True."
    )


def test_rollout_doc_table_reflects_true():
    """La tabla de knobs del doc de rollout no debe quedar stale en `False`."""
    doc = (_BACKEND_ROOT / "docs" / "macro_rollout_and_validation.md").read_text(encoding="utf-8")
    row = next(
        (ln for ln in doc.splitlines() if "`MEALFIT_MACRO_SOLVER_ENABLED`" in ln and "|" in ln),
        None,
    )
    assert row is not None, "No se encontró la fila de MEALFIT_MACRO_SOLVER_ENABLED en la tabla del doc."
    assert "`True`" in row, (
        f"La fila del doc de rollout para MEALFIT_MACRO_SOLVER_ENABLED debería decir `True`: {row!r}"
    )
