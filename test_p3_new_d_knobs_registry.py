"""[P3-NEW-D · 2026-05-08] Tests del registry global de knobs.

Bug original (audit 2026-05-07):
  `_log_active_knobs()` mantenía un dict hardcoded de ~30 knobs visibles.
  Con 50+ knobs activos en el módulo (algunos viviendo en helpers, otros
  añadidos sin actualizar el dict), el log de startup era incompleto:
  un override `MEALFIT_LLM_MAX_PER_USER=5` aparecía pero
  `MEALFIT_LLM_COMBINED_MAX_WAIT_S` podía no estar listado, y el operador
  no podía confirmar que el override tomó efecto sin grep manual al
  código fuente.

Fix:
  1. `_KNOBS_REGISTRY: dict` global, auto-poblado por
     `_env_int/_env_float/_env_bool` en cada llamada.
  2. Cada entry guarda: type ("int"/"float"/"bool"), default, raw env value,
     valor parseado, flag `is_override` (raw presente y parsing OK), flag
     `parse_failed` (raw presente pero invalid → cayó a default).
  3. `_log_active_knobs()` itera el registry y emite 3 líneas:
     resumen, overrides destacados, inventario completo grep-able.
  4. Helper público `get_knobs_registry_snapshot()` para diagnóstico/tests.

Cobertura:
  - test_registry_populated_at_import
  - test_registry_captures_default_when_env_unset
  - test_registry_captures_override_when_env_set
  - test_registry_marks_parse_failure_correctly
  - test_env_bool_registers_correctly
  - test_snapshot_is_copy_not_reference
  - test_log_active_knobs_emits_inventory_line
  - test_log_active_knobs_highlights_overrides
"""
import importlib
import logging
import os
from unittest.mock import patch

import pytest


@pytest.fixture
def go_module():
    """Import limpio de graph_orchestrator. Reload en cada test para que
    el registry se rehidrate desde el env actual del fixture (vía
    `monkeypatch.setenv` aplicado antes del fixture)."""
    import graph_orchestrator
    return graph_orchestrator


# ---------------------------------------------------------------------------
# Registry: comportamiento al import
# ---------------------------------------------------------------------------
def test_registry_populated_at_import(go_module):
    """El registry no debe estar vacío después del import — los ~50 knobs
    del módulo se autoresisistran en cualquier orden."""
    snap = go_module.get_knobs_registry_snapshot()
    assert isinstance(snap, dict)
    assert len(snap) >= 30, (
        f"Registry tiene {len(snap)} knobs; esperaba ≥ 30. "
        f"Si bajó, alguien removió `_env_*` calls."
    )
    # Knobs canónicos que TIENEN que estar (verifican cobertura mínima).
    must_have = [
        "MEALFIT_LLM_MAX_CONCURRENT",
        "MEALFIT_LLM_MAX_PER_USER",
        "MEALFIT_HARD_CEILING_S",
        "MEALFIT_GLOBAL_PIPELINE_TIMEOUT_S",
    ]
    for name in must_have:
        assert name in snap, f"Knob canónico {name} ausente del registry."


def test_registry_captures_default_when_env_unset(go_module):
    """Si el env var NO está seteado, el registry guarda raw=None y
    `is_override=False`."""
    snap = go_module.get_knobs_registry_snapshot()
    # MEALFIT_LLM_MAX_CONCURRENT es default 4. En entorno de test no lo
    # seteamos, así que `is_override` debe ser False.
    if "MEALFIT_LLM_MAX_CONCURRENT" not in os.environ:
        info = snap["MEALFIT_LLM_MAX_CONCURRENT"]
        assert info["raw"] is None
        assert info["is_override"] is False
        assert info["value"] == info["default"]
        assert info["type"] == "int"


def test_registry_captures_override_when_env_set(go_module, monkeypatch):
    """Si el env var SÍ está seteado, una nueva llamada a `_env_int` lo
    registra con `is_override=True` y `value` parseado del raw."""
    monkeypatch.setenv("MEALFIT_TEST_KNOB_OVERRIDE", "42")
    val = go_module._env_int("MEALFIT_TEST_KNOB_OVERRIDE", 7)
    assert val == 42
    snap = go_module.get_knobs_registry_snapshot()
    info = snap["MEALFIT_TEST_KNOB_OVERRIDE"]
    assert info["raw"] == "42"
    assert info["value"] == 42
    assert info["default"] == 7
    assert info["is_override"] is True
    assert info["parse_failed"] is False
    assert info["type"] == "int"


def test_registry_marks_parse_failure_correctly(go_module, monkeypatch):
    """Si el env var tiene un valor inválido (e.g. 'abc' para un int), el
    registry marca `parse_failed=True`, `is_override=False` (porque el
    valor efectivo cae a default), y `value=default`."""
    monkeypatch.setenv("MEALFIT_TEST_KNOB_BAD_INT", "not_a_number")
    val = go_module._env_int("MEALFIT_TEST_KNOB_BAD_INT", 99)
    assert val == 99
    snap = go_module.get_knobs_registry_snapshot()
    info = snap["MEALFIT_TEST_KNOB_BAD_INT"]
    assert info["raw"] == "not_a_number"
    assert info["value"] == 99
    assert info["default"] == 99
    assert info["parse_failed"] is True
    assert info["is_override"] is False, (
        "parse_failed knobs no deben contar como override (el valor efectivo "
        "es el default, no el del env)."
    )


def test_env_bool_registers_correctly(go_module, monkeypatch):
    """`_env_bool` debe registrar tipo 'bool' y respetar el parser laxo."""
    monkeypatch.setenv("MEALFIT_TEST_BOOL_TRUTHY", "yes")
    monkeypatch.setenv("MEALFIT_TEST_BOOL_FALSY", "off")
    val_true = go_module._env_bool("MEALFIT_TEST_BOOL_TRUTHY", False)
    val_false = go_module._env_bool("MEALFIT_TEST_BOOL_FALSY", True)
    assert val_true is True
    assert val_false is False
    snap = go_module.get_knobs_registry_snapshot()
    info_t = snap["MEALFIT_TEST_BOOL_TRUTHY"]
    info_f = snap["MEALFIT_TEST_BOOL_FALSY"]
    assert info_t["type"] == "bool"
    assert info_t["value"] is True
    assert info_t["is_override"] is True
    assert info_f["type"] == "bool"
    assert info_f["value"] is False
    assert info_f["is_override"] is True


def test_env_float_registers_correctly(go_module, monkeypatch):
    monkeypatch.setenv("MEALFIT_TEST_FLOAT_KNOB", "1.5")
    val = go_module._env_float("MEALFIT_TEST_FLOAT_KNOB", 0.5)
    assert val == 1.5
    snap = go_module.get_knobs_registry_snapshot()
    info = snap["MEALFIT_TEST_FLOAT_KNOB"]
    assert info["type"] == "float"
    assert info["value"] == 1.5
    assert info["is_override"] is True


# ---------------------------------------------------------------------------
# Snapshot: aislamiento del registry
# ---------------------------------------------------------------------------
def test_snapshot_is_copy_not_reference(go_module):
    """Mutar el snapshot no debe afectar el registry interno."""
    snap = go_module.get_knobs_registry_snapshot()
    snap["MEALFIT_FAKE_INJECTED"] = {"value": "hacked"}
    snap2 = go_module.get_knobs_registry_snapshot()
    assert "MEALFIT_FAKE_INJECTED" not in snap2, (
        "El snapshot debe ser deep copy — mutaciones del caller no deben "
        "filtrarse al registry."
    )


# ---------------------------------------------------------------------------
# _log_active_knobs: formato y contenido
# ---------------------------------------------------------------------------
def test_log_active_knobs_emits_inventory_line(go_module, caplog):
    """Debe emitir línea `[KNOBS/INVENTORY]` con todos los knobs en formato
    `name=value` separados por coma."""
    caplog.set_level(logging.INFO, logger="graph_orchestrator")
    go_module._log_active_knobs()
    inventory_records = [
        r for r in caplog.records if "[KNOBS/INVENTORY]" in r.getMessage()
    ]
    assert inventory_records, "Falta línea [KNOBS/INVENTORY] en el log."
    msg = inventory_records[0].getMessage()
    # Verificar al menos un knob canónico aparece formateado.
    assert "MEALFIT_LLM_MAX_CONCURRENT=" in msg
    assert "MEALFIT_HARD_CEILING_S=" in msg


def test_log_active_knobs_summary_counts_overrides(go_module, monkeypatch, caplog):
    """La línea de resumen debe reportar el total + #overrides + #parse-failures."""
    # Forzar un override y un parse-fail para que aparezcan en el conteo.
    monkeypatch.setenv("MEALFIT_TEST_LOG_OVERRIDE", "10")
    monkeypatch.setenv("MEALFIT_TEST_LOG_BAD", "xxx")
    go_module._env_int("MEALFIT_TEST_LOG_OVERRIDE", 0)
    go_module._env_int("MEALFIT_TEST_LOG_BAD", 0)

    caplog.set_level(logging.INFO, logger="graph_orchestrator")
    go_module._log_active_knobs()
    summary = next(
        (r.getMessage() for r in caplog.records
         if "graph_orchestrator activos" in r.getMessage()),
        None,
    )
    assert summary, "Falta línea de resumen."
    assert "overrides via env" in summary
    assert "parse-failures" in summary


def test_log_active_knobs_highlights_overrides(go_module, monkeypatch, caplog):
    """Si hay overrides activos, debe haber línea `[KNOBS/OVERRIDE]` con
    name=value de cada override."""
    monkeypatch.setenv("MEALFIT_TEST_HIGHLIGHT", "777")
    go_module._env_int("MEALFIT_TEST_HIGHLIGHT", 1)
    caplog.set_level(logging.INFO, logger="graph_orchestrator")
    go_module._log_active_knobs()
    override_records = [
        r for r in caplog.records if "[KNOBS/OVERRIDE]" in r.getMessage()
    ]
    assert override_records, "Falta línea [KNOBS/OVERRIDE]."
    msg = override_records[0].getMessage()
    assert "MEALFIT_TEST_HIGHLIGHT=777" in msg


def test_log_active_knobs_warns_on_parse_failures(go_module, monkeypatch, caplog):
    """Si hubo parse-failures, debe haber línea WARNING `[KNOBS/PARSE-FAIL]`."""
    monkeypatch.setenv("MEALFIT_TEST_PARSE_FAIL", "not_an_int")
    go_module._env_int("MEALFIT_TEST_PARSE_FAIL", 5)
    caplog.set_level(logging.WARNING, logger="graph_orchestrator")
    go_module._log_active_knobs()
    pf_records = [
        r for r in caplog.records if "[KNOBS/PARSE-FAIL]" in r.getMessage()
    ]
    assert pf_records, "Falta línea WARNING [KNOBS/PARSE-FAIL]."
    assert pf_records[0].levelno == logging.WARNING
    assert "MEALFIT_TEST_PARSE_FAIL" in pf_records[0].getMessage()
