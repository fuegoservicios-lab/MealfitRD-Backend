"""[P2-A · 2026-05-07] Tests del consumer de `_shopping_coherence_block` en
`review_plan_node`.

Bug original (re-audit 2026-05-07):
  `MEALFIT_SHOPPING_COHERENCE_GUARD=block` seteaba
  `plan_result["_shopping_coherence_block"]` con divergencias críticas pero
  ningún consumer downstream leía ese flag → el plan se persistía igual que
  en mode `warn`. Contradicción de contrato: "block" no bloqueaba.

Fix:
  `review_plan_node` ahora consume el flag (mismo patrón que `_schema_invalid`
  y `_recipe_coherence_errors`). La acción se modula por
  `MEALFIT_SHOPPING_COHERENCE_BLOCK_ACTION`:
    - `reject_minor` (default): severity=minor → retry si budget permite.
    - `reject_high`: severity=high (regenerable) → retry forzado por
      `_classify_high_severity` (no matchea keywords contextuales).
    - `degrade`: kill switch — limpia flag, no-op.
    - inválido → reject_minor + warning de knob.

Cobertura:
  - sin flag → no-op (sanity, garantiza que no rompemos el path normal)
  - flag con default action → reject_minor + approved=False + issues populated
  - flag con action=reject_high → severity=high
  - flag con action=degrade → flag stripped, no rejection
  - flag con action=invalid → reject_minor + warning
  - flag + schema_invalid → severity stays critical (max preservado)
  - issue text → `_classify_high_severity` returns "regenerable"
"""
import asyncio
import logging

import pytest

import graph_orchestrator
from graph_orchestrator import _classify_high_severity


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _bypass_form_data():
    """Form data sin restricciones — `review_plan_node` bypassa el LLM/fact-check
    y va directo a las validaciones deterministas (donde vive nuestro consumer)."""
    return {
        "user_id": "guest",  # bypass también el persist de rejection_patterns
        "allergies": [],
        "medicalConditions": [],
        "dislikes": [],
        "dietType": "balanced",
        "_days_to_generate": 3,
    }


def _minimal_plan(*, with_block=None, with_schema_invalid=False):
    """Plan mínimo que pasa schema validation (or not, si with_schema_invalid).

    [2026-06-29] Macros EN BANDA (delivered == target, band_score=1.0) para que el gate ortogonal
    P2-BAND-RETRY-GATE (añadido 2026-06-21, default ON) NO interfiera: sin macros el band_score era 0.0 y
    el gate rechazaba/elevaba severidad, contaminando la prueba del consumer del `_shopping_coherence_block`
    (este test es del CONSUMER, no de la banda). 150p·4 + 200c·4 + 67f·9 ≈ 2003 kcal (coherente)."""
    plan = {
        "calories": 2000,
        "macros": {"protein": 150, "carbs": 200, "fats": 67},
        "days": [
            {"day": 1, "meals": [
                {"meal": "almuerzo", "name": "Pollo con arroz", "ingredients": ["200 g pollo", "150 g arroz"],
                 "protein": 150, "carbs": 200, "fats": 67, "cals": 2000}
            ]}
        ],
    }
    if with_block is not None:
        plan["_shopping_coherence_block"] = with_block
    if with_schema_invalid:
        plan["_schema_invalid"] = True
        plan["_schema_errors"] = "synthetic schema fail"
    return plan


def _minimal_state(*, plan_result):
    """State mínimo para llamar `review_plan_node` con LLM bypass + sin DB writes."""
    return {
        "plan_result": plan_result,
        "form_data": _bypass_form_data(),
        "taste_profile": "",
        "attempt": 1,
        "rejection_reasons": [],
        "_rejection_severity": "minor",
        "request_id": "test-p2-a",
    }


def _run(coro):
    """Ejecuta una corrutina sin pytest-asyncio (no es dependencia del repo)."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# 1. Sanity: sin flag → no-op (no degrada el path normal)
# ---------------------------------------------------------------------------
def test_no_block_flag_no_change(monkeypatch):
    """Sin `_shopping_coherence_block`, el consumer no debe afectar el flujo.
    Un plan limpio con bypass de restricciones debe aprobarse."""
    monkeypatch.delenv("MEALFIT_SHOPPING_COHERENCE_BLOCK_ACTION", raising=False)
    state = _minimal_state(plan_result=_minimal_plan(with_block=None))
    result = _run(graph_orchestrator.review_plan_node(state))
    assert result["review_passed"] is True, \
        "Plan limpio + bypass restrictions debería aprobarse; el consumer P2-A no debe interferir"


# ---------------------------------------------------------------------------
# 2. Default action `reject_minor` cuando flag está presente
# ---------------------------------------------------------------------------
def test_default_action_reject_minor(monkeypatch):
    """Flag presente + sin env var → action=reject_minor → approved=False,
    severity=minor, issues incluye el sample de foods."""
    monkeypatch.delenv("MEALFIT_SHOPPING_COHERENCE_BLOCK_ACTION", raising=False)
    block = [
        {"food": "Cilantro", "side": "expected_only"},
        {"food": "Pollo", "magnitude": True, "expected_qty": 1000, "actual_qty": 500},
    ]
    state = _minimal_state(plan_result=_minimal_plan(with_block=block))
    result = _run(graph_orchestrator.review_plan_node(state))
    assert result["review_passed"] is False
    assert result["_rejection_severity"] == "minor"
    issues = result["rejection_reasons"]
    assert any("COHERENCIA RECETAS LISTA" in i for i in issues), issues
    assert any("Cilantro" in i for i in issues), issues


# ---------------------------------------------------------------------------
# 3. action=reject_high → severity=high
# ---------------------------------------------------------------------------
def test_action_reject_high_elevates_severity(monkeypatch):
    monkeypatch.setenv("MEALFIT_SHOPPING_COHERENCE_BLOCK_ACTION", "reject_high")
    block = [{"food": "Cilantro", "side": "expected_only"}]
    state = _minimal_state(plan_result=_minimal_plan(with_block=block))
    result = _run(graph_orchestrator.review_plan_node(state))
    assert result["review_passed"] is False
    assert result["_rejection_severity"] == "high"


# ---------------------------------------------------------------------------
# 4. action=degrade → flag stripped, no rejection (kill switch)
# ---------------------------------------------------------------------------
def test_action_degrade_clears_flag_no_rejection(monkeypatch):
    monkeypatch.setenv("MEALFIT_SHOPPING_COHERENCE_BLOCK_ACTION", "degrade")
    block = [{"food": "Cilantro", "side": "expected_only"}]
    plan = _minimal_plan(with_block=block)
    state = _minimal_state(plan_result=plan)
    result = _run(graph_orchestrator.review_plan_node(state))
    # Plan se aprueba (sin restricciones + flag limpiado)
    assert result["review_passed"] is True, \
        "degrade kill switch debe permitir que el plan se apruebe"
    # Flag fue removido del plan in-place
    assert "_shopping_coherence_block" not in plan, \
        "degrade debe limpiar el flag para evitar persistirlo"


# ---------------------------------------------------------------------------
# 5. action inválido → fallback a reject_minor + warning
# ---------------------------------------------------------------------------
def test_action_invalid_falls_back_to_reject_minor(monkeypatch, caplog):
    monkeypatch.setenv("MEALFIT_SHOPPING_COHERENCE_BLOCK_ACTION", "garbage_value")
    block = [{"food": "Cilantro", "side": "expected_only"}]
    state = _minimal_state(plan_result=_minimal_plan(with_block=block))
    with caplog.at_level(logging.WARNING):
        result = _run(graph_orchestrator.review_plan_node(state))
    assert result["review_passed"] is False
    assert result["_rejection_severity"] == "minor"
    # Warning de knob inválido debe aparecer
    assert any(
        "MEALFIT_SHOPPING_COHERENCE_BLOCK_ACTION" in r.getMessage()
        and "garbage_value" in r.getMessage()
        for r in caplog.records
    ), [r.getMessage() for r in caplog.records]


# ---------------------------------------------------------------------------
# 6. Severity max preservada: schema_invalid (critical) + coherence_block
#    no debe degradar critical → high/minor.
# ---------------------------------------------------------------------------
def test_critical_severity_preserved_when_block_present(monkeypatch):
    """`_severity_max` debe garantizar que un schema_invalid (critical) no
    sea degradado por el consumer del coherence_block."""
    monkeypatch.setenv("MEALFIT_SHOPPING_COHERENCE_BLOCK_ACTION", "reject_high")
    block = [{"food": "Cilantro", "side": "expected_only"}]
    plan = _minimal_plan(with_block=block, with_schema_invalid=True)
    state = _minimal_state(plan_result=plan)
    result = _run(graph_orchestrator.review_plan_node(state))
    assert result["review_passed"] is False
    assert result["_rejection_severity"] == "critical", \
        "schema_invalid critical no debe ser degradado por coherence_block"


# ---------------------------------------------------------------------------
# 7. Issue text del consumer debe clasificar como "regenerable" para que
#    el retry forzado por reject_high funcione end-to-end.
# ---------------------------------------------------------------------------
def test_issue_text_classified_as_regenerable():
    """`_classify_high_severity` no debe matchear ningún keyword contextual
    (despensa/alergia/condición) sobre nuestro mensaje. Si lo hiciera, el
    `should_retry` cortaría el retry pensando que es no-recuperable."""
    issue_text = (
        "COHERENCIA RECETAS LISTA: 2 divergencia(s) críticas "
        "(foods: Cilantro, Pollo). action=reject_high."
    )
    classification = _classify_high_severity([issue_text])
    assert classification == "regenerable", \
        f"El mensaje del consumer debe ser 'regenerable' para que reject_high " \
        f"realmente dispare retry. Got {classification!r}."


# ---------------------------------------------------------------------------
# 8. Empty list (`[]`) en el flag NO debe disparar el consumer.
# ---------------------------------------------------------------------------
def test_empty_block_list_no_op(monkeypatch):
    """`plan["_shopping_coherence_block"] = []` (caso defensivo: el guard
    nunca debería persistir un set vacío, pero si por algún refactor ocurre,
    el consumer no debe disparar reject)."""
    monkeypatch.setenv("MEALFIT_SHOPPING_COHERENCE_BLOCK_ACTION", "reject_high")
    state = _minimal_state(plan_result=_minimal_plan(with_block=[]))
    result = _run(graph_orchestrator.review_plan_node(state))
    assert result["review_passed"] is True, \
        "Lista vacía no debe disparar rejection (consumer guard)"
