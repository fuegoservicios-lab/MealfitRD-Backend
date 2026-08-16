"""[CRITICAL-1 · post-review-final] Tests del consumer de `_culinary_contract_violations`
(capa 1, P1-CULINARY-CONTRACT) y `_culinary_judge_history`/violations (capa 2, P1-CULINARY-
JUDGE) en `review_plan_node`.

Bug original (review whole-branch, 19 commits, post-P1-CULINARY-JUDGE):
  Ambos bloques `if ... and CULINARY_*_GUARD == "block":` hacían `issues.append(...)` +
  `severity = _severity_max(...)` pero JAMÁS `approved = False`. El veredicto final del nodo
  (`if approved: ... else: ...`) no releía `issues`/`severity` para decidir — solo el flag
  `approved` importaba. Resultado: `MEALFIT_CULINARY_CONTRACT_GUARD=block` (y su gemelo del
  juez) NO rechazaban NADA — el plan se aprobaba igual que en modo `warn`, con las violaciones
  acumuladas en `issues` pero descartadas en la rama `if approved:`. Misma clase de bug que
  P1-G (`_shopping_coherence_block` sin consumer) — ver `test_p2_a_shopping_coherence_block_
  enforcement.py`, cuya técnica este archivo espeja: `review_plan_node` corre COMPLETO (sin
  mockear el nodo), con el LLM bypasseado vía form_data sin restricciones, y el catálogo/juez
  mockeados para producir exactamente 1 violación determinista.

Fix:
  `approved = False` añadido en AMBOS bloques (graph_orchestrator.py, capa 1 ~L38517 y capa 2
  ~L38562), espejando el patrón ya usado por `_shopping_coherence_block` (~L38441).

Vía del test funcional elegida (documentada en el reporte del fixwave):
  Réplica DIRECTA de la técnica del test hermano sobre `review_plan_node` en vivo — no
  extracción a un helper `_apply_culinary_gate`. `culinary_contract_scan`/`get_master_
  ingredients` (capa 1) y `run_culinary_judge` (capa 2) son ambos triviales de monkeypatchear
  sin DB real (el primero es puro — módulo `culinary_coherence.py` sin env/LLM/DB — el segundo
  es una función async top-level en `graph_orchestrator`), así que la técnica del hermano SÍ
  resultó replicable tras leer ambos módulos. El bug es una línea faltante en dos sitios
  estructuralmente idénticos al patrón ya probado; extraer un helper para dos call-sites de 2
  líneas cada uno habría sido sobre-ingeniería.
"""
import asyncio

import pytest

import graph_orchestrator
import shopping_calculator


# ---------------------------------------------------------------------------
# Helpers (espejo de test_p2_a_shopping_coherence_block_enforcement.py)
# ---------------------------------------------------------------------------
def _bypass_form_data():
    """Form data sin restricciones — `review_plan_node` bypassa el LLM/fact-check
    y va directo a las validaciones deterministas (donde viven los gates culinarios)."""
    return {
        "user_id": "guest",  # bypass también el persist de rejection_patterns
        "allergies": [],
        "medicalConditions": [],
        "dislikes": [],
        "dietType": "balanced",
        "_days_to_generate": 3,
    }


def _minimal_plan():
    """Plan mínimo que pasa schema validation + el dish-quality gate (3 pilares + tiempo,
    P1-OBJECTIVE-V4-BATCH) y trae macros EN BANDA (band_score=1.0, P2-BAND-RETRY-GATE no
    interfiere). El paso "El Toque de Fuego" hierve el Casabe — Casabe es ready_to_eat en el
    catálogo sintético de abajo, así que ese paso SIEMPRE produce exactamente 1 violación V1
    determinista vía `culinary_contract_scan` (capa 1)."""
    return {
        "calories": 2000,
        "macros": {"protein": 150, "carbs": 200, "fats": 67},
        "days": [
            {"day": 1, "meals": [
                {"meal": "almuerzo", "name": "Casabe con pollo",
                 "ingredients": ["100 g Casabe", "150 g Pechuga de pollo"],
                 "recipe": [
                     "Mise en place: pesa el casabe y la pechuga.",
                     "El Toque de Fuego: hierve el casabe 5 min y cocina la pechuga 10 min.",
                     "Montaje: sirve el casabe junto a la pechuga.",
                 ],
                 "protein": 150, "carbs": 200, "fats": 67, "cals": 2000}
            ]}
        ],
    }


def _minimal_state(*, plan_result):
    return {
        "plan_result": plan_result,
        "form_data": _bypass_form_data(),
        "taste_profile": "",
        "attempt": 1,
        "rejection_reasons": [],
        "_rejection_severity": "minor",
        "request_id": "test-critical-1",
    }


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


_SYNTHETIC_CATALOG = [
    {"name": "Casabe", "prep_methods": ["tostar", "ninguno"], "ready_to_eat": True},
    {"name": "Pechuga de pollo",
     "prep_methods": ["hervir", "plancha", "freir", "hornear", "guisar", "saltear"],
     "ready_to_eat": False},
]


# ---------------------------------------------------------------------------
# Capa 1 — P1-CULINARY-CONTRACT (scan determinista)
# ---------------------------------------------------------------------------
def test_contract_guard_off_no_rejection(monkeypatch):
    """Sanity: guard=off (irrelevante si hay violaciones) no debe interferir."""
    monkeypatch.setattr(graph_orchestrator, "CULINARY_CONTRACT_GUARD", "off")
    monkeypatch.setattr(graph_orchestrator, "CULINARY_JUDGE_GUARD", "off")
    state = _minimal_state(plan_result=_minimal_plan())
    result = _run(graph_orchestrator.review_plan_node(state))
    assert result["review_passed"] is True


def test_contract_guard_warn_detects_but_does_not_reject(monkeypatch):
    """warn (default de producción): la violación se detecta y queda en
    `_culinary_contract_violations`, pero el plan se APRUEBA (telemetría sin poder)."""
    monkeypatch.setattr(graph_orchestrator, "CULINARY_CONTRACT_GUARD", "warn")
    monkeypatch.setattr(graph_orchestrator, "CULINARY_JUDGE_GUARD", "off")
    monkeypatch.setattr(shopping_calculator, "get_master_ingredients",
                         lambda: _SYNTHETIC_CATALOG)
    plan = _minimal_plan()
    state = _minimal_state(plan_result=plan)
    result = _run(graph_orchestrator.review_plan_node(state))
    assert result["review_passed"] is True, (
        "warn no debe rechazar — es telemetría pura hasta la escalada a block")
    assert plan.get("_culinary_contract_violations"), (
        "el scan debe poblar _culinary_contract_violations aunque no rechace"
    )


def test_contract_guard_block_rejects_plan(monkeypatch):
    """[CRITICAL-1] block CON 1 violación V1 determinista → el plan se RECHAZA
    (approved=False, review_passed=False, rejection_reasons no vacío). Pre-fix,
    esta aserción fallaba: `issues.append` corría pero `approved` seguía True."""
    monkeypatch.setattr(graph_orchestrator, "CULINARY_CONTRACT_GUARD", "block")
    monkeypatch.setattr(graph_orchestrator, "CULINARY_JUDGE_GUARD", "off")
    monkeypatch.setattr(shopping_calculator, "get_master_ingredients",
                         lambda: _SYNTHETIC_CATALOG)
    state = _minimal_state(plan_result=_minimal_plan())
    result = _run(graph_orchestrator.review_plan_node(state))
    assert result["review_passed"] is False, (
        "CULINARY_CONTRACT_GUARD=block con violaciones DEBE rechazar el plan — "
        "regresión clase P1-G si esto se aprueba."
    )
    issues = result["rejection_reasons"]
    assert issues, "rejection_reasons no debe quedar vacío en un rechazo"
    assert any("incoherencia culinaria" in i for i in issues), issues
    assert any("Casabe" in i for i in issues), issues


# ---------------------------------------------------------------------------
# Capa 2 — P1-CULINARY-JUDGE (juicio LLM, mockeado)
# ---------------------------------------------------------------------------
def _fake_judge_report_factory(*, severidad="high"):
    # [P1-COUNTRY-SYSTEM-F1 · 2026-08-16] `run_culinary_judge` ganó un 2º parámetro
    # (`country`, default 'DO' — F1-T3); el fake debe aceptarlo aunque no lo use (el
    # comportamiento mockeado no depende del país).
    async def _fake_run_culinary_judge(plan, country="DO"):
        return graph_orchestrator.CulinaryJudgeReport(violations=[
            graph_orchestrator.CulinaryViolation(
                day=1, meal="Almuerzo", tipo="tecnica_impropia",
                detalle="técnica no corresponde al ingrediente", severidad=severidad,
            )
        ])
    return _fake_run_culinary_judge


def test_judge_guard_warn_detects_but_does_not_reject(monkeypatch):
    monkeypatch.setattr(graph_orchestrator, "CULINARY_CONTRACT_GUARD", "off")
    monkeypatch.setattr(graph_orchestrator, "CULINARY_JUDGE_GUARD", "warn")
    monkeypatch.setattr(graph_orchestrator, "run_culinary_judge",
                         _fake_judge_report_factory())
    plan = _minimal_plan()
    state = _minimal_state(plan_result=plan)
    result = _run(graph_orchestrator.review_plan_node(state))
    assert result["review_passed"] is True
    hist = plan.get("_culinary_judge_history")
    assert hist and hist[-1]["action_taken"] == "warn_only", hist


def test_judge_guard_block_rejects_plan(monkeypatch):
    """[CRITICAL-1] Mismo bug, capa 2 (juez LLM): block CON 1 violación mockeada →
    el plan se RECHAZA. Pre-fix el `action_taken` de la history ya decía "blocked"
    pero el veredicto real (`approved`) no lo reflejaba — contradicción entre la
    telemetría y el comportamiento observable."""
    monkeypatch.setattr(graph_orchestrator, "CULINARY_CONTRACT_GUARD", "off")
    monkeypatch.setattr(graph_orchestrator, "CULINARY_JUDGE_GUARD", "block")
    monkeypatch.setattr(graph_orchestrator, "run_culinary_judge",
                         _fake_judge_report_factory())
    plan = _minimal_plan()
    state = _minimal_state(plan_result=plan)
    result = _run(graph_orchestrator.review_plan_node(state))
    assert result["review_passed"] is False, (
        "CULINARY_JUDGE_GUARD=block con violaciones DEBE rechazar el plan."
    )
    issues = result["rejection_reasons"]
    assert issues, "rejection_reasons no debe quedar vacío en un rechazo"
    assert any("tecnica_impropia" in i for i in issues), issues
    hist = plan.get("_culinary_judge_history")
    assert hist and hist[-1]["action_taken"] == "blocked", hist


def test_judge_history_capped_at_20(monkeypatch):
    """[IMPORTANT-4] `_culinary_judge_history` se trunca a [-20:] tras el append,
    mismo patrón que su gemelo `_shopping_coherence_block_history`."""
    monkeypatch.setattr(graph_orchestrator, "CULINARY_CONTRACT_GUARD", "off")
    monkeypatch.setattr(graph_orchestrator, "CULINARY_JUDGE_GUARD", "warn")
    monkeypatch.setattr(graph_orchestrator, "run_culinary_judge",
                         _fake_judge_report_factory())
    plan = _minimal_plan()
    plan["_culinary_judge_history"] = [{"ts": "seed", "model": "x", "violations": [],
                                         "action_taken": "warn_only"} for _ in range(25)]
    state = _minimal_state(plan_result=plan)
    _run(graph_orchestrator.review_plan_node(state))
    assert len(plan["_culinary_judge_history"]) == 20, len(plan["_culinary_judge_history"])
