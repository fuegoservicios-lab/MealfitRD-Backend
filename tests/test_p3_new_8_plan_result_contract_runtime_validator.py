"""[P3-NEW-8 · 2026-05-11] `_ensure_plan_result_contract` runtime validator.

Tests:
    A) Behavioral: validador detecta los 5 modos de fallo de tipo/rango.
    B) Parser-based: helper se invoca al return final del pipeline.
"""
from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_GRAPH_FP = _REPO_ROOT / "backend" / "graph_orchestrator.py"

sys.path.insert(0, str(_REPO_ROOT / "backend"))


@pytest.fixture(scope="module")
def src() -> str:
    return _GRAPH_FP.read_text(encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────
# A) Behavioral — el validador detecta drift de tipo/rango.
# ──────────────────────────────────────────────────────────────────────


def test_validator_silent_on_well_formed(caplog):
    from graph_orchestrator import _ensure_plan_result_contract
    caplog.set_level(logging.WARNING)
    pr = {
        "days": [{"meals": []}],
        "_review_severity": "minor",
        "_is_fallback": False,
        "_review_failed_but_delivered": True,
        "_review_issues": ["x"],
    }
    _ensure_plan_result_contract(pr, source="test_well_formed")
    p3_warnings = [r for r in caplog.records if "P3-NEW-8" in r.message]
    assert not p3_warnings, (
        "P3-NEW-8 regresión: validador loguea warning sobre plan well-formed: "
        f"{[r.message for r in p3_warnings]}"
    )


def test_validator_detects_invalid_severity(caplog):
    from graph_orchestrator import _ensure_plan_result_contract
    caplog.set_level(logging.WARNING)
    pr = {"_review_severity": "medium"}  # fuera del set canónico
    _ensure_plan_result_contract(pr, source="test_invalid_severity")
    assert any(
        "_review_severity" in r.message and "medium" in r.message
        for r in caplog.records
    ), "P3-NEW-8 regresión: severity inválido no detectado"


def test_validator_detects_non_bool_is_fallback(caplog):
    from graph_orchestrator import _ensure_plan_result_contract
    caplog.set_level(logging.WARNING)
    pr = {"_is_fallback": "true"}  # string, no bool
    _ensure_plan_result_contract(pr, source="test_non_bool")
    assert any(
        "_is_fallback" in r.message and "no es bool" in r.message
        for r in caplog.records
    ), "P3-NEW-8 regresión: _is_fallback no-bool no detectado"


def test_validator_detects_non_list_days(caplog):
    from graph_orchestrator import _ensure_plan_result_contract
    caplog.set_level(logging.WARNING)
    pr = {"days": {"day0": {}}}  # dict, no list
    _ensure_plan_result_contract(pr, source="test_non_list_days")
    assert any(
        "days" in r.message and "no es lista" in r.message
        for r in caplog.records
    ), "P3-NEW-8 regresión: days no-list no detectado"


def test_validator_handles_none_input(caplog):
    """plan_result=None no debe crashear — solo loguear."""
    from graph_orchestrator import _ensure_plan_result_contract
    caplog.set_level(logging.WARNING)
    _ensure_plan_result_contract(None, source="test_none")
    # Debería haber loguear "plan_result=None"
    assert any(
        "plan_result=None" in r.message for r in caplog.records
    )


def test_validator_handles_non_dict_input(caplog):
    """plan_result no-dict no debe crashear — solo loguear."""
    from graph_orchestrator import _ensure_plan_result_contract
    caplog.set_level(logging.WARNING)
    _ensure_plan_result_contract([1, 2, 3], source="test_list")
    assert any(
        "plan_result no es dict" in r.message for r in caplog.records
    )


def test_validator_does_not_mutate_input():
    """Sin side-effects sobre el dict."""
    from graph_orchestrator import _ensure_plan_result_contract
    pr_original = {
        "_review_severity": "minor",
        "_is_fallback": False,
        "days": [{"x": 1}],
    }
    pr_copy = dict(pr_original)
    _ensure_plan_result_contract(pr_original, source="test_no_mutation")
    assert pr_original == pr_copy, (
        "P3-NEW-8 regresión: el validador mutó plan_result. DEBE ser "
        "side-effect-free."
    )


# ──────────────────────────────────────────────────────────────────────
# B) Parser-based — invocación al return final del pipeline.
# ──────────────────────────────────────────────────────────────────────


def test_validator_invoked_at_pipeline_return(src: str):
    """`_ensure_plan_result_contract` debe invocarse antes del
    `return plan_to_return` final de arun_plan_pipeline."""
    # Boundary: arun_plan_pipeline → return plan_to_return.
    arun_idx = src.find("async def arun_plan_pipeline(")
    assert arun_idx > 0
    ret_idx = src.find("\n        return plan_to_return", arun_idx)
    assert ret_idx > 0, "return plan_to_return no encontrado"

    # En las ~30 líneas previas al return, debe estar la invocación.
    pre = src[max(0, ret_idx - 2000):ret_idx]
    assert "_ensure_plan_result_contract(plan_to_return" in pre, (
        "P3-NEW-8 regresión: `_ensure_plan_result_contract` ya no se "
        "invoca antes del return final de arun_plan_pipeline. Sin "
        "invocación, el validador es código muerto."
    )
    # Debe estar dentro de try/except (best-effort, NEVER raise).
    assert "try:" in pre and "except Exception" in pre, (
        "P3-NEW-8 regresión: la invocación del validador ya no está "
        "envuelta en try/except. Una excepción aquí abortaría el "
        "return del pipeline — peor que el drift que detecta."
    )


def test_validator_function_defined(src: str):
    """La función debe estar definida (sanity check)."""
    assert "def _ensure_plan_result_contract(plan_result" in src, (
        "P3-NEW-8 regresión: `_ensure_plan_result_contract` ya no está "
        "definido en graph_orchestrator.py."
    )
