"""[P1-32] Tests para que `_apply_critical_review_guardrails` detecte y
regenere fallbacks anómalos (`_is_fallback=True` + `_schema_invalid=True`
o rechazo crítico).

Bug original (audit P1-32):
  Pre-fix, el guardrail tenía:

    needs_critical_fallback = (
        isinstance(plan_result, dict) and not already_fallback and (
            schema_invalid
            or (not review_passed and rejection_severity == "critical")
        )
    )

  El clause `not already_fallback` short-circuiteaba TODA la rama de
  fallback. Asumía: "si el plan ya es _is_fallback, no necesitamos
  fallback". Pero si llegamos con `_is_fallback=True` + `_schema_invalid
  =True` (mutación downstream corrupta, doble graceful degradation, bug
  en `_get_extreme_fallback_plan`), el frontend recibía un plan que no
  podía renderizar — hojas en blanco. Para `_is_fallback + critical
  rejection`, el plan se entregaba pese a violar alergias/condiciones
  médicas declaradas — riesgo de salud.

Fix:
  Detectar explícitamente la anomalía (`already_fallback` AND
  (`schema_invalid` OR `critical rejection`)), emitir `logger.error` con
  marker `[P1-32]` para que SRE alerte sobre regresiones de
  `_get_extreme_fallback_plan`, y resetear `already_fallback=False` para
  forzar la regeneración del fallback en la rama estándar.

Cobertura:
  - test_anomalous_schema_invalid_fallback_triggers_regeneration
  - test_anomalous_critical_rejection_fallback_triggers_regeneration
  - test_anomalous_path_logs_error_with_marker
  - test_normal_fallback_still_skipped_when_not_anomalous
  - test_normal_critical_rejection_still_triggers_fallback
  - test_normal_schema_invalid_still_triggers_fallback
  - test_happy_path_no_action
  - test_documentation_p1_32_present
"""
import inspect
import logging
from unittest.mock import patch, MagicMock

import pytest

import graph_orchestrator
from graph_orchestrator import _apply_critical_review_guardrails


_NUTRITION = {
    "target_calories": 2000,
    "total_daily_calories": 2000,
    "macros": {"protein_g": 150, "carbs_g": 200, "fats_g": 60,
               "protein_str": "150g", "carbs_str": "200g", "fats_str": "60g"},
    "total_daily_macros": {"protein_g": 150, "carbs_g": 200, "fats_g": 60,
                           "protein_str": "150g", "carbs_str": "200g", "fats_str": "60g"},
    "goal_label": "mantener",
}
_FORM = {"mainGoal": "mantener"}


def _fresh_fallback_plan(*args, num_days=None, **kwargs):
    """Stub para `_get_extreme_fallback_plan`: devuelve un plan válido
    marcado como fallback. Acepta la signature real
    `(nutrition, mainGoal, *, num_days=N)` vía *args/**kwargs."""
    if num_days is None:
        num_days = kwargs.get("num_days", 7)
    return {
        "days": [{"day": i + 1, "meals": []} for i in range(num_days)],
        "_is_fallback": True,
        "calories": 2000,
        "macros": {"protein": "150g", "carbs": "200g", "fats": "60g"},
    }


# ---------------------------------------------------------------------------
# 1. Anómalos: regeneran fresh fallback.
# ---------------------------------------------------------------------------
def test_anomalous_schema_invalid_fallback_triggers_regeneration():
    """Plan con `_is_fallback=True` + `_schema_invalid=True` debe
    regenerarse — el frontend no puede renderizar el plan corrupto."""
    state = {
        "plan_result": {
            "_is_fallback": True,
            "_schema_invalid": True,
            "_schema_errors": "Day 3 missing 'meals'",
            "days": [],  # corrupt
        },
        "review_passed": True,
        "_rejection_severity": None,
        "rejection_reasons": [],
    }
    with patch.object(graph_orchestrator, "_get_extreme_fallback_plan",
                      side_effect=_fresh_fallback_plan) as mock_fb:
        _apply_critical_review_guardrails(
            state, nutrition=_NUTRITION, actual_form_data=_FORM,
            requested_days=7,
        )

    mock_fb.assert_called_once()
    new_plan = state["plan_result"]
    # El plan nuevo debe tener `_critical_rejection` y disclaimer.
    assert new_plan.get("_critical_rejection") is True
    assert "_review_disclaimer" in new_plan
    assert "estructura inválida" in new_plan["_review_disclaimer"]


def test_anomalous_critical_rejection_fallback_triggers_regeneration():
    """Plan con `_is_fallback=True` + `not review_passed` +
    severity='critical' debe regenerarse — un fallback no debería
    violar alergias declaradas, así que es señal de bug."""
    state = {
        "plan_result": {
            "_is_fallback": True,
            "days": [{"day": 1, "meals": [{"name": "Pollo con maní"}]}],
        },
        "review_passed": False,
        "_rejection_severity": "critical",
        "rejection_reasons": ["Plan contiene maní; usuario alérgico"],
    }
    with patch.object(graph_orchestrator, "_get_extreme_fallback_plan",
                      side_effect=_fresh_fallback_plan) as mock_fb:
        _apply_critical_review_guardrails(
            state, nutrition=_NUTRITION, actual_form_data=_FORM,
            requested_days=3,
        )

    mock_fb.assert_called_once()
    new_plan = state["plan_result"]
    assert new_plan.get("_critical_rejection") is True
    assert new_plan.get("_review_severity") == "critical"
    assert "alergias" in new_plan["_review_disclaimer"]


def test_anomalous_path_logs_error_with_marker(caplog):
    """La detección debe emitir `logger.error` con marker `[P1-32]` para
    que SRE alerte sobre la regresión."""
    state = {
        "plan_result": {
            "_is_fallback": True,
            "_schema_invalid": True,
            "days": [],
        },
        "review_passed": True,
        "_rejection_severity": None,
        "rejection_reasons": [],
    }
    with patch.object(graph_orchestrator, "_get_extreme_fallback_plan",
                      side_effect=_fresh_fallback_plan), \
         caplog.at_level(logging.ERROR, logger="graph_orchestrator"):
        _apply_critical_review_guardrails(
            state, nutrition=_NUTRITION, actual_form_data=_FORM,
            requested_days=7,
        )

    p132_logs = [r for r in caplog.records if "[P1-32]" in r.getMessage()]
    assert p132_logs, (
        f"P1-32: esperaba logger.error con [P1-32]. "
        f"Logs: {[r.getMessage()[:120] for r in caplog.records]}"
    )
    # Severidad ERROR (no warning) — anomalía vs degradación esperada.
    assert p132_logs[0].levelname == "ERROR"


# ---------------------------------------------------------------------------
# 2. Casos normales: comportamiento previo preservado.
# ---------------------------------------------------------------------------
def test_normal_fallback_still_skipped_when_not_anomalous():
    """Plan `_is_fallback=True` + `review_passed=True` + sin schema_invalid
    NO debe regenerarse (path normal post-fallback)."""
    plan = {
        "_is_fallback": True,
        "days": [{"day": 1, "meals": []}],
    }
    state = {
        "plan_result": plan,
        "review_passed": True,
        "_rejection_severity": None,
        "rejection_reasons": [],
    }
    with patch.object(graph_orchestrator, "_get_extreme_fallback_plan",
                      side_effect=_fresh_fallback_plan) as mock_fb:
        _apply_critical_review_guardrails(
            state, nutrition=_NUTRITION, actual_form_data=_FORM,
            requested_days=3,
        )

    mock_fb.assert_not_called()
    # plan original preservado.
    assert state["plan_result"] is plan


def test_normal_critical_rejection_still_triggers_fallback():
    """Plan NO-fallback con rechazo crítico → regenera (path original
    preservado)."""
    state = {
        "plan_result": {
            "days": [{"day": 1, "meals": [{"name": "Pollo con maní"}]}],
        },
        "review_passed": False,
        "_rejection_severity": "critical",
        "rejection_reasons": ["Allergy violation"],
    }
    with patch.object(graph_orchestrator, "_get_extreme_fallback_plan",
                      side_effect=_fresh_fallback_plan) as mock_fb:
        _apply_critical_review_guardrails(
            state, nutrition=_NUTRITION, actual_form_data=_FORM,
            requested_days=3,
        )

    mock_fb.assert_called_once()
    assert state["plan_result"]["_critical_rejection"] is True


def test_normal_schema_invalid_still_triggers_fallback():
    """Plan NO-fallback con `_schema_invalid=True` → regenera."""
    state = {
        "plan_result": {
            "_schema_invalid": True,
            "_schema_errors": "Day shape invalid",
            "days": [],
        },
        "review_passed": False,
        "_rejection_severity": None,
        "rejection_reasons": [],
    }
    with patch.object(graph_orchestrator, "_get_extreme_fallback_plan",
                      side_effect=_fresh_fallback_plan) as mock_fb:
        _apply_critical_review_guardrails(
            state, nutrition=_NUTRITION, actual_form_data=_FORM,
            requested_days=3,
        )

    mock_fb.assert_called_once()
    assert "estructura inválida" in state["plan_result"]["_review_disclaimer"]


def test_happy_path_no_action():
    """Plan normal + review_passed=True + sin issues → no-op."""
    plan = {
        "days": [{"day": 1, "meals": []}],
        "calories": 2000,
    }
    state = {
        "plan_result": plan,
        "review_passed": True,
        "_rejection_severity": None,
        "rejection_reasons": [],
    }
    with patch.object(graph_orchestrator, "_get_extreme_fallback_plan",
                      side_effect=_fresh_fallback_plan) as mock_fb:
        _apply_critical_review_guardrails(
            state, nutrition=_NUTRITION, actual_form_data=_FORM,
            requested_days=3,
        )

    mock_fb.assert_not_called()
    assert state["plan_result"] is plan
    # No mutaciones sobre el plan.
    assert "_review_failed_but_delivered" not in plan
    assert "_critical_rejection" not in plan


def test_non_critical_rejection_marks_banner_only():
    """Rechazo NO-crítico → no regenera, marca para banner ámbar."""
    plan = {
        "days": [{"day": 1, "meals": []}],
    }
    state = {
        "plan_result": plan,
        "review_passed": False,
        "_rejection_severity": "minor",
        "rejection_reasons": ["Diversidad de proteínas baja"],
    }
    with patch.object(graph_orchestrator, "_get_extreme_fallback_plan",
                      side_effect=_fresh_fallback_plan) as mock_fb:
        _apply_critical_review_guardrails(
            state, nutrition=_NUTRITION, actual_form_data=_FORM,
            requested_days=3,
        )

    mock_fb.assert_not_called()
    assert plan.get("_review_failed_but_delivered") is True
    assert plan.get("_review_severity") == "minor"


# ---------------------------------------------------------------------------
# 3. Documentación.
# ---------------------------------------------------------------------------
def test_documentation_p1_32_present():
    """Comentario `[P1-32]` debe documentar el rationale."""
    src = inspect.getsource(_apply_critical_review_guardrails)
    assert "[P1-32]" in src, (
        "P1-32: falta marker en _apply_critical_review_guardrails."
    )


def test_documentation_mentions_anomaly_or_corruption():
    """El comentario debe explicar el rationale: anomalía / corrupción
    / regresión de _get_extreme_fallback_plan / no debería ocurrir."""
    src = inspect.getsource(_apply_critical_review_guardrails)
    p132_idx = src.find("[P1-32]")
    window = src[p132_idx : p132_idx + 2500]
    needles = ["anómal", "anomal", "corrupt", "regresión", "regression",
               "no debería", "no deberia", "doble", "mutación", "race"]
    found = any(n.lower() in window.lower() for n in needles)
    assert found, (
        f"P1-32: el comentario debe explicar la naturaleza anómala. "
        f"Encontrado: {window[:300]!r}"
    )
