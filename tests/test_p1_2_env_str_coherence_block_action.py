"""[P1-2 · 2026-05-08] Tests del helper `_env_str` y migración del knob
`MEALFIT_SHOPPING_COHERENCE_BLOCK_ACTION` al registry.

Bug original (audit 2026-05-07):
  `review_plan_node` leía el knob con `os.environ.get(...).strip().lower()`
  directo, saltándose `_env_*` y por tanto el `_KNOBS_REGISTRY`. Como
  consecuencia:
    1. El knob no aparecía en el log de inventario `[KNOBS/INVENTORY]`.
    2. `/api/system/knobs` (que iteran `_KNOBS_REGISTRY`) lo invisibilizaba.
    3. La validación de choices vivía duplicada inline en review_plan_node.

Fix:
  1. Nuevo helper `_env_str(name, default, choices=None)` que normaliza
     (strip+lower), valida contra `choices` opcional y registra.
  2. Llamada inline reemplazada por `_env_str("...", "reject_minor",
     choices={"degrade","reject_minor","reject_high"})`.
  3. Comportamiento end-user idéntico: valor válido → usado; inválido →
     WARNING + default `reject_minor`. Pero ahora visible en inventario.

Cobertura:
  - _env_str con default cuando env vacío.
  - _env_str con override válido.
  - _env_str con choices inválido → default + parse_failed=True + WARNING.
  - Knob aparece en `_KNOBS_REGISTRY` con type="str".
  - Snapshot incluye el knob tras el primer call.
"""
import importlib
import logging
import os
import sys

import pytest


@pytest.fixture(autouse=True)
def _reset_env_and_module():
    """Cada test arranca con env limpio y orchestrator re-importable."""
    prev = os.environ.pop("MEALFIT_SHOPPING_COHERENCE_BLOCK_ACTION", None)
    prev_test = os.environ.pop("MEALFIT_P1_2_TEST_KNOB", None)
    yield
    if prev is not None:
        os.environ["MEALFIT_SHOPPING_COHERENCE_BLOCK_ACTION"] = prev
    if prev_test is not None:
        os.environ["MEALFIT_P1_2_TEST_KNOB"] = prev_test


def _fresh_orchestrator():
    if "graph_orchestrator" in sys.modules:
        del sys.modules["graph_orchestrator"]
    import graph_orchestrator
    return graph_orchestrator


def test_env_str_returns_default_when_unset():
    go = _fresh_orchestrator()
    val = go._env_str("MEALFIT_P1_2_TEST_KNOB", "default_val")
    assert val == "default_val"


def test_env_str_returns_normalized_override():
    os.environ["MEALFIT_P1_2_TEST_KNOB"] = "  REJECT_HIGH  "
    go = _fresh_orchestrator()
    val = go._env_str(
        "MEALFIT_P1_2_TEST_KNOB",
        "reject_minor",
        choices={"degrade", "reject_minor", "reject_high"},
    )
    assert val == "reject_high"


def test_env_str_invalid_choice_falls_back_with_warning(caplog):
    os.environ["MEALFIT_P1_2_TEST_KNOB"] = "bogus_value"
    go = _fresh_orchestrator()
    with caplog.at_level(logging.WARNING):
        val = go._env_str(
            "MEALFIT_P1_2_TEST_KNOB",
            "reject_minor",
            choices={"degrade", "reject_minor", "reject_high"},
        )
    assert val == "reject_minor"
    assert any("MEALFIT_P1_2_TEST_KNOB" in rec.message and "no es valor permitido" in rec.message
               for rec in caplog.records)


def test_env_str_no_choices_accepts_any_value():
    os.environ["MEALFIT_P1_2_TEST_KNOB"] = "FREE_FORM"
    go = _fresh_orchestrator()
    val = go._env_str("MEALFIT_P1_2_TEST_KNOB", "x")
    assert val == "free_form"


def test_env_str_registers_in_registry_with_type_str():
    go = _fresh_orchestrator()
    go._env_str("MEALFIT_P1_2_TEST_KNOB", "x", choices={"x", "y"})
    snap = go.get_knobs_registry_snapshot()
    assert "MEALFIT_P1_2_TEST_KNOB" in snap
    entry = snap["MEALFIT_P1_2_TEST_KNOB"]
    assert entry["type"] == "str"
    assert entry["default"] == "x"
    assert entry["value"] == "x"
    assert entry["is_override"] is False
    assert entry["parse_failed"] is False


def test_env_str_invalid_value_marks_parse_failed():
    os.environ["MEALFIT_P1_2_TEST_KNOB"] = "invalid"
    go = _fresh_orchestrator()
    go._env_str("MEALFIT_P1_2_TEST_KNOB", "x", choices={"x", "y"})
    snap = go.get_knobs_registry_snapshot()
    entry = snap["MEALFIT_P1_2_TEST_KNOB"]
    assert entry["parse_failed"] is True
    assert entry["value"] == "x"
    assert entry["raw"] == "invalid"


def test_coherence_block_action_knob_appears_in_registry_after_use():
    """El knob real debe aparecer en el registry tras el primer call de review_plan_node.

    Verificamos llamando _env_str directo con el nombre real (lo que hace
    el call site). No invocamos review_plan_node entero porque requiere
    state setup pesado; el invariante es que el helper auto-registra.
    """
    go = _fresh_orchestrator()
    val = go._env_str(
        "MEALFIT_SHOPPING_COHERENCE_BLOCK_ACTION",
        "reject_minor",
        choices={"degrade", "reject_minor", "reject_high"},
    )
    assert val == "reject_minor"
    snap = go.get_knobs_registry_snapshot()
    assert "MEALFIT_SHOPPING_COHERENCE_BLOCK_ACTION" in snap
    assert snap["MEALFIT_SHOPPING_COHERENCE_BLOCK_ACTION"]["type"] == "str"


def test_coherence_block_action_override_visible_in_overrides():
    os.environ["MEALFIT_SHOPPING_COHERENCE_BLOCK_ACTION"] = "degrade"
    go = _fresh_orchestrator()
    go._env_str(
        "MEALFIT_SHOPPING_COHERENCE_BLOCK_ACTION",
        "reject_minor",
        choices={"degrade", "reject_minor", "reject_high"},
    )
    snap = go.get_knobs_registry_snapshot()
    entry = snap["MEALFIT_SHOPPING_COHERENCE_BLOCK_ACTION"]
    assert entry["is_override"] is True
    assert entry["value"] == "degrade"
    assert entry["raw"] == "degrade"


def test_call_site_uses_env_str_not_raw_environ():
    """Smoke: confirma que el patrón raw `os.environ.get("MEALFIT_SHOPPING_COHERENCE_BLOCK_ACTION")`
    ya no aparece en graph_orchestrator.py (queda solo el call por _env_str)."""
    import pathlib
    # [P1-A · 2026-05-08] Fix path tras P3-CANDIDATE-B: tests viven en
    # `backend/tests/`, no en `backend/`. Subir un nivel para localizar
    # graph_orchestrator.py.
    src = pathlib.Path(__file__).resolve().parent.parent / "graph_orchestrator.py"
    content = src.read_text(encoding="utf-8")
    # No debe haber lectura raw del knob.
    assert 'os.environ.get("MEALFIT_SHOPPING_COHERENCE_BLOCK_ACTION")' not in content
    assert "os.environ.get('MEALFIT_SHOPPING_COHERENCE_BLOCK_ACTION')" not in content
    # Debe haber al menos un _env_str con el nombre del knob.
    assert '_env_str(' in content
    assert '"MEALFIT_SHOPPING_COHERENCE_BLOCK_ACTION"' in content
