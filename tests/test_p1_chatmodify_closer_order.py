"""[P1-CHATMODIFY-CLOSER-ORDER · 2026-07-01] (audit micros GAP-1 / paridad GAP-3)

En chat-modify el micro-closer corría DESPUÉS de computar+asignar las `aggregated_shopping_list*`
y del coherence guard → sus escalados (hasta 1.6× de un ingrediente en cualquier día) quedaban en
la receta persistida pero NO en la lista, y el guard (que existe para medir ese drift) ya había
corrido → divergencia receta↔lista invisible, sin `_coherence_warnings` ni telemetría.

Fix en dos mitades: (1) el closer corre sobre la copia local ANTES de computar las listas (las
listas nacen closer-aware); (2) el callback re-corre el MISMO closer determinista sobre el fresh
ANTES de asignar listas y correr el guard → el guard mide el estado FINAL.
"""
from __future__ import annotations

from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
_TOOLS = (_BACKEND / "tools.py").read_text(encoding="utf-8")


def _modify_body() -> str:
    i = _TOOLS.find("def execute_modify_single_meal")
    assert i != -1
    j = _TOOLS.find("\n@tool", i)
    return _TOOLS[i:j if j != -1 else len(_TOOLS)]


def test_pre_lists_closer_runs_before_list_computation():
    body = _modify_body()
    i_pre = body.find("_cmg_pre(plan_data")
    i_lists = body.find("get_shopping_list_delta(")
    assert i_pre != -1, "falta el closer pre-listas sobre la copia local (P1-CHATMODIFY-CLOSER-ORDER)"
    assert i_lists != -1
    assert i_pre < i_lists, "el closer pre-listas debe correr ANTES de computar las aggregated lists"


def test_callback_closer_before_lists_and_guard():
    body = _modify_body()
    i_cb = body.find("def _apply_meal_modification")
    cb = body[i_cb:]
    i_closer = cb.find("_close_micro_gaps_for_plan(plan_data_fresh")
    i_assign = cb.find('plan_data_fresh["aggregated_shopping_list"]')
    i_guard = cb.find("run_shopping_coherence_guard_and_append_history")
    assert i_closer != -1 and i_assign != -1 and i_guard != -1, "faltan callsites en el callback"
    assert i_closer < i_assign < i_guard, (
        "orden requerido en el callback: closer → asignación de listas → coherence guard "
        "(el guard debe medir el estado FINAL)"
    )


def test_no_closer_after_guard_in_callback():
    body = _modify_body()
    i_cb = body.find("def _apply_meal_modification")
    cb = body[i_cb:]
    i_guard = cb.find("run_shopping_coherence_guard_and_append_history")
    after_guard = cb[i_guard:]
    assert "_close_micro_gaps_for_plan(" not in after_guard, (
        "regresión: el closer volvió a correr DESPUÉS del guard → drift receta↔lista invisible"
    )


def test_marker_present():
    assert "P1-CHATMODIFY-CLOSER-ORDER" in _TOOLS
    # el anchor histórico del recompute de micros se preserva (bloque movido, no eliminado).
    assert "P2-CHATMODIFY-MICROS-STALE" in _TOOLS
