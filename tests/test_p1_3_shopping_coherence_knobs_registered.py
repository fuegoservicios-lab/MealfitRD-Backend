"""[P1-3 · 2026-05-08] Tests de registry para knobs cross-module en shopping_calculator.

Bug original (audit 2026-05-07):
  Dos knobs leídos directos con `os.environ.get` en `backend/shopping_calculator.py`:
    - `MEALFIT_SHOPPING_COHERENCE_GUARD` (línea ~2392, choices off/warn/block)
    - `MEALFIT_SHOPPING_COHERENCE_TOLERANCE_PCT` (línea ~2411, float (0,1))
  Como el helper `_env_*` vive en graph_orchestrator.py, estos saltaban el
  `_KNOBS_REGISTRY` (P3-NEW-D) y eran invisibles en `[KNOBS/INVENTORY]` y en
  `/api/system/knobs`. Misma sintomatología que P1-2 pero en otro módulo.

Fix:
  1. `_env_float` extendido con `validator: Callable[[float], bool] | None`
     para validación de rango (mantiene API uniforme con `_env_str(choices=...)`).
  2. Call sites en shopping_calculator.py migrados a lazy import del orchestrator
     (mismo patrón que db_inventory.py:104 — evita coupling top-level).
  3. Defensivo: try/except con fallback al comportamiento legacy si el
     orchestrator no se puede importar (test isolation, circular boot).

Cobertura:
  - _env_float con validator que retorna True → registra normal.
  - _env_float con validator que retorna False → default + WARNING + parse_failed.
  - _env_float con validator que lanza excepción → default + parse_failed.
  - Knob `_GUARD` aparece en registry tras llamada a `_get_coherence_guard_mode`.
  - Knob `_TOLERANCE_PCT` aparece en registry tras llamada con override válido.
  - Override out-of-range en `_TOLERANCE_PCT` → registry marca parse_failed.
  - Smoke: ningún `os.environ.get("MEALFIT_SHOPPING_COHERENCE_*")` raw fuera del fallback try/except.
"""
import logging
import os
import sys

import pytest


@pytest.fixture(autouse=True)
def _reset_env():
    keys = [
        "MEALFIT_SHOPPING_COHERENCE_GUARD",
        "MEALFIT_SHOPPING_COHERENCE_TOLERANCE_PCT",
        "MEALFIT_P1_3_TEST_FLOAT",
    ]
    snap = {k: os.environ.pop(k, None) for k in keys}
    yield
    for k, v in snap.items():
        if v is not None:
            os.environ[k] = v


def _fresh_modules():
    for m in ("graph_orchestrator", "shopping_calculator"):
        if m in sys.modules:
            del sys.modules[m]


def test_env_float_validator_passes_when_in_range():
    _fresh_modules()
    import graph_orchestrator as go
    os.environ["MEALFIT_P1_3_TEST_FLOAT"] = "0.25"
    val = go._env_float("MEALFIT_P1_3_TEST_FLOAT", 0.10, validator=lambda v: 0.0 < v < 1.0)
    assert val == 0.25
    snap = go.get_knobs_registry_snapshot()
    assert snap["MEALFIT_P1_3_TEST_FLOAT"]["parse_failed"] is False
    assert snap["MEALFIT_P1_3_TEST_FLOAT"]["is_override"] is True


def test_env_float_validator_rejects_out_of_range_with_warning(caplog):
    _fresh_modules()
    import graph_orchestrator as go
    os.environ["MEALFIT_P1_3_TEST_FLOAT"] = "1.5"
    with caplog.at_level(logging.WARNING):
        val = go._env_float("MEALFIT_P1_3_TEST_FLOAT", 0.10, validator=lambda v: 0.0 < v < 1.0)
    assert val == 0.10
    snap = go.get_knobs_registry_snapshot()
    assert snap["MEALFIT_P1_3_TEST_FLOAT"]["parse_failed"] is True
    assert snap["MEALFIT_P1_3_TEST_FLOAT"]["value"] == 0.10
    assert any("fuera de rango" in rec.message for rec in caplog.records)


def test_env_float_validator_exception_falls_back_to_default(caplog):
    _fresh_modules()
    import graph_orchestrator as go
    os.environ["MEALFIT_P1_3_TEST_FLOAT"] = "0.5"

    def _bad_validator(v):
        raise RuntimeError("boom")

    with caplog.at_level(logging.WARNING):
        val = go._env_float("MEALFIT_P1_3_TEST_FLOAT", 0.10, validator=_bad_validator)
    assert val == 0.10
    snap = go.get_knobs_registry_snapshot()
    assert snap["MEALFIT_P1_3_TEST_FLOAT"]["parse_failed"] is True


def test_env_float_no_validator_keeps_legacy_behavior():
    _fresh_modules()
    import graph_orchestrator as go
    os.environ["MEALFIT_P1_3_TEST_FLOAT"] = "999.0"
    val = go._env_float("MEALFIT_P1_3_TEST_FLOAT", 0.10)
    assert val == 999.0
    snap = go.get_knobs_registry_snapshot()
    assert snap["MEALFIT_P1_3_TEST_FLOAT"]["parse_failed"] is False


def test_coherence_guard_mode_registers_in_orchestrator_registry():
    _fresh_modules()
    import shopping_calculator as sc
    val = sc._get_coherence_guard_mode()
    assert val == "warn"
    import graph_orchestrator as go
    snap = go.get_knobs_registry_snapshot()
    assert "MEALFIT_SHOPPING_COHERENCE_GUARD" in snap
    assert snap["MEALFIT_SHOPPING_COHERENCE_GUARD"]["type"] == "str"
    assert snap["MEALFIT_SHOPPING_COHERENCE_GUARD"]["default"] == "warn"


def test_coherence_guard_mode_invalid_choice_falls_back():
    os.environ["MEALFIT_SHOPPING_COHERENCE_GUARD"] = "bogus"
    _fresh_modules()
    import shopping_calculator as sc
    val = sc._get_coherence_guard_mode()
    assert val == "warn"
    import graph_orchestrator as go
    snap = go.get_knobs_registry_snapshot()
    assert snap["MEALFIT_SHOPPING_COHERENCE_GUARD"]["parse_failed"] is True


def test_coherence_guard_mode_valid_override():
    os.environ["MEALFIT_SHOPPING_COHERENCE_GUARD"] = "block"
    _fresh_modules()
    import shopping_calculator as sc
    val = sc._get_coherence_guard_mode()
    assert val == "block"
    import graph_orchestrator as go
    snap = go.get_knobs_registry_snapshot()
    assert snap["MEALFIT_SHOPPING_COHERENCE_GUARD"]["is_override"] is True
    assert snap["MEALFIT_SHOPPING_COHERENCE_GUARD"]["value"] == "block"


def test_coherence_tolerance_pct_registers_in_registry():
    _fresh_modules()
    import shopping_calculator as sc
    val = sc._get_coherence_tolerance_pct()
    assert val == 0.10
    import graph_orchestrator as go
    snap = go.get_knobs_registry_snapshot()
    assert "MEALFIT_SHOPPING_COHERENCE_TOLERANCE_PCT" in snap
    assert snap["MEALFIT_SHOPPING_COHERENCE_TOLERANCE_PCT"]["type"] == "float"
    assert snap["MEALFIT_SHOPPING_COHERENCE_TOLERANCE_PCT"]["default"] == 0.10


def test_coherence_tolerance_pct_valid_override():
    os.environ["MEALFIT_SHOPPING_COHERENCE_TOLERANCE_PCT"] = "0.25"
    _fresh_modules()
    import shopping_calculator as sc
    val = sc._get_coherence_tolerance_pct()
    assert val == 0.25
    import graph_orchestrator as go
    snap = go.get_knobs_registry_snapshot()
    assert snap["MEALFIT_SHOPPING_COHERENCE_TOLERANCE_PCT"]["is_override"] is True
    assert snap["MEALFIT_SHOPPING_COHERENCE_TOLERANCE_PCT"]["parse_failed"] is False


def test_coherence_tolerance_pct_out_of_range_falls_back():
    os.environ["MEALFIT_SHOPPING_COHERENCE_TOLERANCE_PCT"] = "1.5"
    _fresh_modules()
    import shopping_calculator as sc
    val = sc._get_coherence_tolerance_pct()
    assert val == 0.10
    import graph_orchestrator as go
    snap = go.get_knobs_registry_snapshot()
    assert snap["MEALFIT_SHOPPING_COHERENCE_TOLERANCE_PCT"]["parse_failed"] is True
    assert snap["MEALFIT_SHOPPING_COHERENCE_TOLERANCE_PCT"]["value"] == 0.10


def test_coherence_tolerance_pct_zero_rejected():
    os.environ["MEALFIT_SHOPPING_COHERENCE_TOLERANCE_PCT"] = "0"
    _fresh_modules()
    import shopping_calculator as sc
    val = sc._get_coherence_tolerance_pct()
    assert val == 0.10


def test_coherence_tolerance_pct_invalid_string_falls_back():
    os.environ["MEALFIT_SHOPPING_COHERENCE_TOLERANCE_PCT"] = "abc"
    _fresh_modules()
    import shopping_calculator as sc
    val = sc._get_coherence_tolerance_pct()
    assert val == 0.10


def test_no_raw_environ_get_outside_fallback_in_shopping_calculator():
    """Smoke: la lectura raw del knob no debe aparecer en `shopping_calculator.py`.

    [P2-1 · 2026-05-08] Tras extraer los helpers a `backend/knobs.py` (cero
    deps internas), el patrón try/except con fallback `os.environ.get` dejó de
    hacer falta. Los call sites usan directamente `_knob_env_str` /
    `_knob_env_float` importados a top-level. Cualquier raw read reintroducido
    es regresión.
    """
    import pathlib, re
    # [P1-A · 2026-05-08] Tras P3-CANDIDATE-B (mv tests al subdir `tests/`),
    # `__file__.parent` ya NO es `backend/`. Subir un nivel para localizar el
    # módulo bajo test (mismo patrón que test_p1_a_knobs_registry_extended).
    src = pathlib.Path(__file__).resolve().parent.parent / "shopping_calculator.py"
    content = src.read_text(encoding="utf-8")
    # Contar occurrences de cada knob raw
    guard_raw = len(re.findall(r'os\.environ\.get\(\s*["\']MEALFIT_SHOPPING_COHERENCE_GUARD["\']', content))
    tol_raw = len(re.findall(r'os\.environ\.get\(\s*["\']MEALFIT_SHOPPING_COHERENCE_TOLERANCE_PCT["\']', content))
    # Cero reads raw post-P2-1: el helper SSOT es la única ruta.
    assert guard_raw == 0, (
        f"GUARD raw env reads = {guard_raw} (esperado 0 post-P2-1; usa _knob_env_str)."
    )
    assert tol_raw == 0, (
        f"TOLERANCE_PCT raw env reads = {tol_raw} (esperado 0 post-P2-1; usa _knob_env_float)."
    )
