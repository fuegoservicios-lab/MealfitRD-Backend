"""[P1-BARIATRIC-CRITICAL-RETRY · 2026-06-28] FASE 4 del roadmap "motor LLM-suficiente". HALLAZGO DECISIVO del workflow:
hoy un rechazo CRÍTICO bariátrico hace should_retry → "end" INMEDIATO = CERO reintentos (los "4 intentos" de los logs eran
4 generaciones independientes muriendo al primer crítico). Fix: en review_plan_node, un crítico bariátrico de ELECCIÓN DE
COMIDA (porción/volumen/grano-crudo/especia/azúcar) se degrada a 'high' regenerable (espejo del precedente DM2 glucémico)
→ should_retry da 1-2 retries quirúrgicos con feedback antes del fallback CURADO. El revisor RE-GATEA cada retry → cero
pérdida de autoridad clínica. Excluye allergen/renal/diet/schema (por flag) + embarazo/mercurio/celíaca/interacción (marca).

Tests PUROS de la cadena lógica (los gate-conditions con flags _had_*_critical viven en review_plan_node, cubiertos
estructuralmente por los tests DM2). Aquí: los predicados del gate + que un 'high' regenerable bariátrico SÍ reintenta.
"""
from __future__ import annotations

import time

import graph_orchestrator as g

_BAR = {"medicalConditions": ["Cirugía Bariátrica (manga gástrica)"]}
_NON = {"medicalConditions": ["Diabetes tipo 2"]}

_FOOD_ISSUE = ["Día 2 - Cena: Batata mediana entera (153 g) excede el límite seguro recomendado de ≤100 g para "
               "paciente bariátrico.", "Día 2 - Cena: 3 huevos en una sola comida es una porción excesiva."]
_RAW_GRAIN = ["Día 2 - Almuerzo: Cebada perlada cruda está CONTRAINDICADA por riesgo de bezoar."]


def test_is_bariatric_condition():
    assert g._is_bariatric_condition(_BAR) is True
    assert g._is_bariatric_condition(_NON) is False
    assert g._is_bariatric_condition({}) is False


def test_food_choice_critical_is_regenerable():
    assert g._critical_is_bariatric_regenerable(_FOOD_ISSUE) is True
    assert g._critical_is_bariatric_regenerable(_RAW_GRAIN) is True  # grano crudo → el LLM puede cocer/sustituir


def test_nonregenerable_safety_not_downgraded():
    # concerns contextuales que NO cambian entre intentos → NO degradar (van a fallback)
    assert g._critical_is_bariatric_regenerable(["Contiene mercurio (pez espada) en embarazo"]) is False
    assert g._critical_is_bariatric_regenerable(["Interacción con warfarina (vitamina K)"]) is False
    assert g._critical_is_bariatric_regenerable([]) is False  # sin señal → conservador


def test_food_issue_classifies_regenerable():
    # tras el downgrade a 'high', should_retry clasifica las razones de comida como regenerable (no contextual)
    assert g._classify_high_severity(_FOOD_ISSUE) == "regenerable"
    assert g._classify_high_severity(_RAW_GRAIN) == "regenerable"


def _state(severity, reasons, attempt=1):
    return {
        "review_passed": False,
        "_rejection_severity": severity,
        "rejection_reasons": reasons,
        "attempt": attempt,
        "pipeline_start": time.time(),  # budget completo
        "plan_result": {"days": [{"meals": []}]},
        "_marker_regen_attempted": True,  # evita la rama marker_regen
    }


def test_high_regenerable_bariatric_retries(monkeypatch):
    # una vez degradado a 'high', un crítico bariátrico de comida REINTENTA (no "end")
    monkeypatch.setattr(g, "MAX_ATTEMPTS", 3)
    assert g.should_retry(_state("high", _FOOD_ISSUE, attempt=1)) == "retry"


def test_critical_still_ends_in_should_retry():
    # should_retry NO hace el downgrade (eso es review_plan_node); un crítico sin degradar sigue → end
    assert g.should_retry(_state("critical", _FOOD_ISSUE, attempt=1)) == "end"


def test_max_attempts_stops_retry(monkeypatch):
    # agotados los intentos, ni siquiera un 'high' regenerable reintenta (→ end → fallback curado)
    monkeypatch.setattr(g, "MAX_ATTEMPTS", 3)
    assert g.should_retry(_state("high", _FOOD_ISSUE, attempt=3)) == "end"


def test_knob_and_anchor():
    import pathlib
    src = pathlib.Path(g.__file__).read_text(encoding="utf-8")
    assert "P1-BARIATRIC-CRITICAL-RETRY" in src
    assert "BARIATRIC_CRITICAL_SOFT_REJECT" in src
    assert g.BARIATRIC_CRITICAL_SOFT_REJECT is True
    # el gate va como elif del gate DM2 (encadenado, no doble-procesa comórbido)
    assert "_critical_is_bariatric_regenerable(issues)" in src
