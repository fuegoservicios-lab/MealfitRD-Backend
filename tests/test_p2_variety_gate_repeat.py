"""[P2-VARIETY-GATE-REPEAT · 2026-06-25] Gate de retry por fruta/plato-base repetido el mismo día.

Observado en el plan 18544d16 (renovación post-fix): el detector `fruit_repeats` funcionaba
(flaggeó lechosa ×2 el Día 1) pero el lever era SOLO el prompt (advisory) → el LLM no obedeció y
el plan se entregó con lechosa en 2 comidas + revoltillo ×2. Este gate convierte la fruta-repetida
(default ON) y el plato-base-repetido (default OFF — falso-positivo con 'plancha'/'ensalada') en
motivos de RECHAZO → `should_retry` fuerza 1 retry acotado con directiva de diversificación.

Helper puro `_variety_repeat_gate_issues(variety_report) -> list[str]`. Knobs
`MEALFIT_VARIETY_GATE_FRUIT_REPEAT` / `MEALFIT_VARIETY_GATE_BASE_DISH_REPEAT`.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import graph_orchestrator as go
from graph_orchestrator import _variety_repeat_gate_issues


def test_fruta_repetida_dispara_rechazo_por_default():
    assert go.VARIETY_GATE_FRUIT_REPEAT is True
    issues = _variety_repeat_gate_issues({"fruit_repeats": 1, "same_day_repeats": 0})
    assert len(issues) == 1
    assert "FRUTA REPETIDA" in issues[0]


def test_sin_repeticion_no_rechaza():
    issues = _variety_repeat_gate_issues({"fruit_repeats": 0, "same_day_repeats": 0})
    assert issues == []


def test_plato_base_repetido_off_por_default():
    """same_day_repeats>0 NO rechaza por default (evita falso-positivo con plancha/ensalada)."""
    assert go.VARIETY_GATE_BASE_DISH_REPEAT is False
    issues = _variety_repeat_gate_issues({"fruit_repeats": 0, "same_day_repeats": 2})
    assert issues == []


def test_plato_base_repetido_on_si_se_habilita(monkeypatch):
    monkeypatch.setattr(go, "VARIETY_GATE_BASE_DISH_REPEAT", True)
    issues = _variety_repeat_gate_issues({"fruit_repeats": 0, "same_day_repeats": 1})
    assert any("PLATO-BASE REPETIDO" in i for i in issues)


def test_fruit_gate_off_no_rechaza(monkeypatch):
    monkeypatch.setattr(go, "VARIETY_GATE_FRUIT_REPEAT", False)
    issues = _variety_repeat_gate_issues({"fruit_repeats": 3, "same_day_repeats": 0})
    assert issues == []


def test_entrada_invalida_no_revienta():
    assert _variety_repeat_gate_issues(None) == []
    assert _variety_repeat_gate_issues("no-dict") == []
    assert _variety_repeat_gate_issues({}) == []


def test_gate_cableado_en_review_plan_node():
    """El gate debe invocarse dentro del bloque VARIETY_HARD_GATE de review_plan_node
    (parser-based: protege contra que alguien borre el cableado dejando el helper huérfano)."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(here, "graph_orchestrator.py"), encoding="utf-8").read()
    assert "_variety_repeat_gate_issues(_vr)" in src, "el gate no está cableado en review_plan_node"
    assert "P2-VARIETY-GATE-REPEAT" in src


def test_knobs_registrados():
    from knobs import get_knobs_registry_snapshot
    snap = get_knobs_registry_snapshot()
    assert "MEALFIT_VARIETY_GATE_FRUIT_REPEAT" in snap
    assert "MEALFIT_VARIETY_GATE_BASE_DISH_REPEAT" in snap
