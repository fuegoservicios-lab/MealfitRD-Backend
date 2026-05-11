"""[P2-AUDIT-3 · 2026-05-10] Test E2E del pipeline COMPLETO de
`_shopping_coherence_block` (producer → consumer).

Bug que protege:
    P1-G original (re-audit 2026-05-07): `MEALFIT_SHOPPING_COHERENCE_GUARD=block`
    persistía el flag pero `review_plan_node` no lo consumía. Tests
    pre-existentes:
      - `test_p1_shopping_recipe_coherence.py` ejercita el PRODUCER
        (`run_shopping_coherence_guard`) — verifica que el flag se setea.
      - `test_p2_a_shopping_coherence_block_enforcement.py` ejercita el
        CONSUMER (`review_plan_node`) — verifica que un flag manualmente
        construido se procesa.
      - `test_p2_2_action_taken_invariant.py` usa regex sobre source.

    Gap del audit 2026-05-10:
      Ningún test ejercita la CADENA producer→consumer end-to-end. Un
      refactor que cambie el shape del flag (ej. dict vs list, nuevo
      campo requerido) podría:
        - Pasar el productor (test_p1 verifica con su shape esperado).
        - Pasar el consumer (test_p2_a construye el shape manualmente).
        - PERO romper la integración (productor emite shape X, consumer
          espera shape Y).

Fix:
    Este test:
      1. Llama al PRODUCTOR real `run_shopping_coherence_guard` con un
         plan_result que tiene divergencia conocida (recipe→list).
      2. Verifica que `_shopping_coherence_block` quedó set en plan_result.
      3. Pasa ese plan_result (sin tocar) al CONSUMER real `review_plan_node`.
      4. Verifica que la acción se aplicó:
         - `_shopping_coherence_block_history[-1].action_taken` hidratado.
         - severity/approved correctos.
         - flag popped si action=degrade.

Cobertura:
    - action=reject_minor (default): severity='minor', approved=False,
      flag persiste, history hidratado.
    - action=reject_high: severity='high', approved=False.
    - action=degrade: flag popped, approved=True (kill switch).
"""
from __future__ import annotations

import asyncio

import pytest

import shopping_calculator
import graph_orchestrator
from shopping_calculator import run_shopping_coherence_guard


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def no_master_db(monkeypatch):
    """Stub master_map para no requerir DB durante el test del productor."""
    monkeypatch.setattr(shopping_calculator, "get_master_ingredients", lambda: [])


def _bypass_form_data():
    """form_data sin restricciones para que el reviewer LLM bypassee."""
    return {
        "user_id": "guest",
        "allergies": [],
        "medicalConditions": [],
        "dislikes": [],
        "dietType": "balanced",
        "_days_to_generate": 3,
    }


def _plan_with_recipe_only_food():
    """Plan con divergencia conocida: receta menciona 'Cilantro' pero
    aggregated_shopping_list NO lo incluye → guard producirá
    `cap_swallowed_modifier` para Cilantro."""
    return {
        "calories": 2000,
        "days": [
            {
                "day": 1,
                "meals": [
                    {
                        "meal": "almuerzo",
                        "name": "Pollo guisado con cilantro",
                        "ingredients_raw": [
                            "200 g pollo",
                            "150 g arroz",
                            "5 g cilantro",
                        ],
                    }
                ],
            }
        ],
        # Lista de compras OMITE cilantro → divergencia presence.
        "aggregated_shopping_list": [
            {"name": "Pollo", "quantity": 200, "unit": "g"},
            {"name": "Arroz", "quantity": 150, "unit": "g"},
        ],
        "calc_household_multiplier": 1.0,
    }


def _state_for_review(plan_result):
    """State mínimo para `review_plan_node` con LLM bypass."""
    return {
        "plan_result": plan_result,
        "form_data": _bypass_form_data(),
        "taste_profile": "",
        "attempt": 1,
        "rejection_reasons": [],
        "_rejection_severity": "minor",
        "request_id": "test-p2-audit-3",
    }


def _run(coro):
    """Ejecuta corrutina sin pytest-asyncio."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# 1. PRODUCTOR setea el flag en plan_result
# ---------------------------------------------------------------------------
def test_producer_sets_coherence_block_flag(monkeypatch):
    """El productor `run_shopping_coherence_guard` con mode='block' debe
    setear `_shopping_coherence_block` en plan_result cuando hay
    divergencia presence."""
    monkeypatch.setenv("MEALFIT_SHOPPING_COHERENCE_GUARD", "block")
    plan = _plan_with_recipe_only_food()
    run_shopping_coherence_guard(plan, mode_override="block", multiplier=1.0)
    assert plan.get("_shopping_coherence_block"), (
        "P2-AUDIT-3 regresión: productor con mode='block' NO setea el "
        "flag pese a divergencia conocida (Cilantro en receta, ausente "
        "en lista). El contrato del productor está roto."
    )
    # El flag debe contener Cilantro como food faltante.
    foods = {d.get("food") for d in plan["_shopping_coherence_block"] if isinstance(d, dict)}
    assert "Cilantro" in foods, (
        f"P2-AUDIT-3: el flag no incluye 'Cilantro' como divergencia. "
        f"Got foods={foods}."
    )


# ---------------------------------------------------------------------------
# 2. CONSUMER procesa el flag DEL PRODUCTOR (no construido manualmente)
# ---------------------------------------------------------------------------
def test_consumer_processes_producer_output_reject_minor(monkeypatch):
    """E2E: productor setea flag → consumer lo lee → action=reject_minor
    (default) → approved=False, severity='minor'."""
    monkeypatch.setenv("MEALFIT_SHOPPING_COHERENCE_GUARD", "block")
    monkeypatch.delenv("MEALFIT_SHOPPING_COHERENCE_BLOCK_ACTION", raising=False)

    plan = _plan_with_recipe_only_food()
    # 1. PRODUCER fase
    run_shopping_coherence_guard(plan, mode_override="block", multiplier=1.0)
    assert plan.get("_shopping_coherence_block"), "Productor no set el flag"

    # 2. CONSUMER fase — pasa plan_result REAL al review_plan_node.
    state = _state_for_review(plan)
    result = _run(graph_orchestrator.review_plan_node(state))

    # 3. Verificaciones del consumer
    assert result["review_passed"] is False, (
        "P2-AUDIT-3 regresión: consumer no rechazó el plan pese a "
        "flag presente. La cadena producer→consumer está rota."
    )
    assert result["_rejection_severity"] == "minor", (
        f"P2-AUDIT-3: severity esperada 'minor' (default reject_minor), "
        f"got {result['_rejection_severity']!r}."
    )
    # Issue text debe mencionar COHERENCIA RECETAS LISTA
    issues = result["rejection_reasons"]
    assert any("COHERENCIA RECETAS LISTA" in i for i in issues), (
        f"P2-AUDIT-3: rejection_reasons no menciona la coherencia. "
        f"Issues={issues}."
    )


def test_consumer_processes_producer_output_reject_high(monkeypatch):
    """E2E con action=reject_high → severity='high'."""
    monkeypatch.setenv("MEALFIT_SHOPPING_COHERENCE_GUARD", "block")
    monkeypatch.setenv("MEALFIT_SHOPPING_COHERENCE_BLOCK_ACTION", "reject_high")

    plan = _plan_with_recipe_only_food()
    run_shopping_coherence_guard(plan, mode_override="block", multiplier=1.0)
    assert plan.get("_shopping_coherence_block")

    state = _state_for_review(plan)
    result = _run(graph_orchestrator.review_plan_node(state))

    assert result["review_passed"] is False
    assert result["_rejection_severity"] == "high", (
        f"P2-AUDIT-3: action=reject_high debe elevar severity a 'high'. "
        f"Got {result['_rejection_severity']!r}."
    )


def test_consumer_degrade_pops_flag_from_producer(monkeypatch):
    """E2E con action=degrade → flag popped, plan se aprueba (kill switch).
    Garantiza que la rama 'degrade' del consumer toca el flag REAL del
    productor (no uno sintético)."""
    monkeypatch.setenv("MEALFIT_SHOPPING_COHERENCE_GUARD", "block")
    monkeypatch.setenv("MEALFIT_SHOPPING_COHERENCE_BLOCK_ACTION", "degrade")

    plan = _plan_with_recipe_only_food()
    run_shopping_coherence_guard(plan, mode_override="block", multiplier=1.0)
    assert "_shopping_coherence_block" in plan, (
        "Productor no set el flag — sanity falló."
    )

    state = _state_for_review(plan)
    result = _run(graph_orchestrator.review_plan_node(state))

    assert result["review_passed"] is True, (
        "P2-AUDIT-3: kill switch 'degrade' debe permitir aprobar el plan."
    )
    assert "_shopping_coherence_block" not in plan, (
        "P2-AUDIT-3: degrade debe POPEAR el flag del plan in-place. "
        "Si persiste, se serializaría a meal_plans innecesariamente."
    )


# ---------------------------------------------------------------------------
# 3. Invariante action_taken hidratado en history
# ---------------------------------------------------------------------------
def test_consumer_hydrates_action_taken_in_history(monkeypatch):
    """Invariante P2-2: tras review_plan_node, el último item de
    `_shopping_coherence_block_history` NO tiene `action_taken=None`.

    Este test es E2E: el productor crea la entry con action_taken=None
    (placeholder), el consumer la debe hidratar con el knob resuelto.
    """
    monkeypatch.setenv("MEALFIT_SHOPPING_COHERENCE_GUARD", "block")
    monkeypatch.setenv("MEALFIT_SHOPPING_COHERENCE_BLOCK_ACTION", "reject_minor")

    plan = _plan_with_recipe_only_food()
    run_shopping_coherence_guard(plan, mode_override="block", multiplier=1.0)
    # NOTE: el history se crea en assemble_plan_node, no en
    # run_shopping_coherence_guard. Aquí seedeamos manualmente la entry
    # con action_taken=None para simular ese estado pre-consumer.
    plan["_shopping_coherence_block_history"] = [
        {
            "ts": "2026-05-10T00:00:00+00:00",
            "attempt": 1,
            "divergence_count": len(plan["_shopping_coherence_block"]),
            "block_set": True,
            "action_taken": None,  # placeholder pre-review
        }
    ]

    state = _state_for_review(plan)
    _run(graph_orchestrator.review_plan_node(state))

    history = plan.get("_shopping_coherence_block_history") or []
    assert history, "P2-AUDIT-3: history desapareció tras review."
    last = history[-1]
    assert isinstance(last, dict), "history[-1] no es dict"
    assert last.get("action_taken") is not None, (
        f"P2-AUDIT-3 regresión: invariante P2-2 rota — `action_taken` "
        f"sigue None tras review_plan_node. history={history}"
    )
    assert last["action_taken"] in {"reject_minor", "reject_high", "degrade", "hydration_error"}, (
        f"P2-AUDIT-3: action_taken con valor inesperado: "
        f"{last['action_taken']!r}"
    )
