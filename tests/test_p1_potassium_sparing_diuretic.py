"""[P1-POTASSIUM-SPARING-DIURETIC · 2026-06-19] (audit fresco P1-2) Ahorradores de potasio ↔ hiperkalemia.

Bug (audit 2026-06-19): MEDICATION_RULES solo modelaba la hiperkalemia para IECA/ARA-II. Los diuréticos
ahorradores de potasio (espironolactona/Aldactone — 1ª línea en HTA resistente/ICC en RD — eplerenona,
amilorida, triamtereno) causan EXACTAMENTE el mismo riesgo (potasio sérico alto → arritmia) pero NO se
detectaban → ni advisory ni gate FS9. Peor: auto-contradictorio, la directiva DASH de la HTA prioriza
potasio (guineo/aguacate/leguminosas/hoja verde) → AMPLIFICABA el riesgo que la fila ace_arb ya advierte.

Fix: nueva fila `potassium_sparing_diuretic` con el mismo riesgo de potasio + prompt_block que toma
PRECEDENCIA explícita sobre el "sube potasio" de DASH; + la fila anticoagulante gana la cláusula simétrica
de vitamina K (DASH prioriza hoja verde mientras la warfarina exige consistencia → la consistencia manda).
"""
from __future__ import annotations

import pytest

import medication_rules as mr


def test_detect_potassium_sparing_variants():
    for med in ("Espironolactona", "espironolactona", "Aldactone", "Eplerenona",
                "Amilorida", "Triamtereno"):
        ids = {r.id for r in mr.detect_active_medications({"medications": [med]})}
        assert "potassium_sparing_diuretic" in ids, med


def test_requires_review_and_advisory():
    fd = {"medications": ["Aldactone"]}
    assert mr.requires_medication_review(fd) is True
    advs = mr.build_medication_advisories(fd)
    assert any("ahorrador" in a["medicamento"].lower() or "espironolactona" in a["medicamento"].lower()
               for a in advs)


def test_prompt_block_takes_precedence_over_dash_potassium():
    out = mr.build_medication_prompt({"medications": ["Espironolactona"]})
    assert "PRECEDENCIA" in out
    assert "DASH" in out
    # Debe instruir a NO maximizar el potasio (lo contrario al riesgo).
    assert "potasio" in out.lower() and ("MODERADAS" in out or "NO maximices" in out)


def test_condition_strings_do_not_falsematch_potassium_sparing():
    # Defensa contra el patrón histórico de falsos-positivos token-substring: ninguna frase de CONDICIÓN
    # (no-medicamento) debe activar la fila potassium_sparing_diuretic via el backstop de texto libre.
    # (Scoped a ESTA fila a propósito — no afirma la lista completa.)
    for cond in ("Resistencia a la insulina", "Insuficiencia renal", "Hipertensión arterial",
                 "Ácido úrico alto", "Hipotiroidismo"):
        ids = {r.id for r in mr.detect_active_medications({"medicalConditions": [cond]})}
        assert "potassium_sparing_diuretic" not in ids, cond


def test_not_anticoagulant_not_timing():
    rule = next(r for r in mr.MEDICATION_RULES if r.id == "potassium_sparing_diuretic")
    assert rule.anticoagulant is False
    assert rule.timing_sensitive is False
    assert rule.precedence == 40  # mismo tier de riesgo de potasio que ace_arb


def test_anticoagulant_block_has_vit_k_dash_clause():
    # Simétrica del crítico: la fila warfarina modera el "más hoja verde" de DASH a "consistente".
    out = mr.build_medication_prompt({"medications": ["Warfarina"]})
    assert "DASH" in out
    assert "CONSISTENTE" in out


# ── Funcional: gate FS9 vía la capa clínica determinista ──
@pytest.fixture(scope="module")
def go():
    import graph_orchestrator as _go
    return _go


def test_fs9_gate_for_potassium_sparing(go):
    plan = {"days": [{"meals": [{"name": "A", "ingredients": ["100g de arroz", "150g de pollo"]}]}]}
    out = go._apply_deterministic_clinical_layer(plan, {"medications": ["Espironolactona"],
                                                        "gender": "female", "age": 55}, {})
    mr_review = out.get("medication_review")
    assert isinstance(mr_review, dict)
    assert any("ahorrador" in lbl.lower() or "espironolactona" in lbl.lower()
               for lbl in mr_review.get("medications", []))
    rpr = out.get("requires_professional_review")
    assert isinstance(rpr, dict) and rpr.get("flag") is True and rpr.get("medication_interaction") is True
