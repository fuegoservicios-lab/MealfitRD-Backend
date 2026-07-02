"""[P2-ENGINE-CLINICAL-SAVERS · 2026-07-02] 2 P2 del test clínico en vivo con Gemini (2026-07-02):
planes clínicamente salvables morían enteros por causas normalizables.

P2-HTA-SALT-NORMALIZE — el revisor rechazó COMPLETO el plan de una hipertensa por:
  (1) "'Sal al gusto' en múltiples comidas" → las subs deterministas de HTA cubrían cubitos/sazón/
      tajín pero NO la sal plana. Nuevas filas en `_HTA_SODIUM_SUBS` (frases completas, jamás 'sal'
      desnuda): "sal al gusto"/"pizca de sal" → "Sal mínima (¼ cdta)". Aplica a HTA y ERC (renal
      reusa el set) en S1 (Guard 3) y updates (backstop) sin cambios extra.
  (2) "enlatados/queso sin especificar bajo en sodio o enjuague" → nota determinista macro-preservante
      (`_apply_low_sodium_canned_notes`, patrón food-safety, knob MEALFIT_HTA_LOWSODIUM_NOTE).

P2-REVIEWER-CB-TRANSIENT — un breaker abierto en el revisor (429s de Gemini) se reportaba como
"Error en la estructura del revisor médico" con severity=critical y ABORT del plan entero: la
Exception genérica del guard no estaba en el whitelist POR TIPO de `_is_reviewer_transient_error`.
Nueva `LLMCircuitOpenError` (subclase de Exception — cero cambios para los `except Exception`)
en los 6 guards de invocación + whitelist. Un CB abierto es transitorio POR DEFINICIÓN.
"""
from __future__ import annotations

from pathlib import Path

import graph_orchestrator as go
from condition_rules import collect_substitutions

_GRAPH = (Path(__file__).resolve().parent.parent / "graph_orchestrator.py").read_text(encoding="utf-8")

_HTA = {"medicalConditions": ["Hipertensión"]}
_ERC = {"medicalConditions": ["Enfermedad renal crónica"]}


# ── P2-HTA-SALT-NORMALIZE · subs de sal ─────────────────────────────────────
def test_salt_tokens_in_hta_and_renal_subs():
    for form in (_HTA, _ERC):
        subs = collect_substitutions(form)
        toks = [t for s in subs for t in s["tokens"]]
        assert "sal al gusto" in toks and "pizca de sal" in toks, \
            f"sal plana sin cubrir para {form['medicalConditions']}"


def test_salt_normalized_in_meal_backstop():
    meal = {"name": "Pollo guisado", "ingredients": ["150g de pollo", "Sal al gusto"],
            "ingredients_raw": ["150g de pollo", "Sal al gusto"],
            "protein": 30, "carbs": 5, "fats": 8, "cals": 220, "recipe": ["El Toque de Fuego: guisa 25 min."]}
    n = go.condition_substitution_backstop_for_meal(meal, _HTA)
    assert n >= 1
    joined = " ".join(str(i) for i in meal["ingredients"]).lower()
    assert "sal al gusto" not in joined, f"sal plana no normalizada: {meal['ingredients']}"
    assert "sal mínima" in joined
    # idempotente (el reemplazo no re-matchea ningún token)
    before = list(meal["ingredients"])
    go.condition_substitution_backstop_for_meal(meal, _HTA)
    assert meal["ingredients"] == before


def test_salt_pinch_and_compound_phrase():
    m1 = {"name": "X", "ingredients": ["1 pizca de sal"], "recipe": []}
    go.condition_substitution_backstop_for_meal(m1, _ERC)
    assert "sal mínima" in " ".join(m1["ingredients"]).lower(), f"pizca sin normalizar: {m1['ingredients']}"
    m2 = {"name": "X", "ingredients": ["Sal y pimienta al gusto"], "recipe": []}
    go.condition_substitution_backstop_for_meal(m2, _HTA)
    j2 = " ".join(m2["ingredients"]).lower()
    assert "sal mínima" in j2 and "pimienta" in j2, f"frase compuesta perdió la pimienta: {m2['ingredients']}"


def test_salt_untouched_without_condition_and_with_negative():
    m = {"name": "X", "ingredients": ["Sal al gusto"], "recipe": []}
    assert go.condition_substitution_backstop_for_meal(m, {"medicalConditions": ["Ninguna"]}) == 0
    assert m["ingredients"] == ["Sal al gusto"]
    m2 = {"name": "X", "ingredients": ["Sal baja en sodio al gusto"], "recipe": []}
    go.condition_substitution_backstop_for_meal(m2, _HTA)
    assert m2["ingredients"] == ["Sal baja en sodio al gusto"], "negative 'baja en sodio' ignorado"


# ── P2-HTA-SALT-NORMALIZE · nota de enlatados ───────────────────────────────
def test_lowsodium_note_added_for_hta_canned():
    plan = {"days": [{"meals": [
        {"name": "Ensalada de atún", "ingredients": ["125g de atún en agua"], "recipe": ["Montaje: sirve."]},
        {"name": "Pollo a la plancha", "ingredients": ["150g de pollo"], "recipe": ["Montaje: sirve."]},
    ]}]}
    n = go._apply_low_sodium_canned_notes(plan, _HTA)
    assert n == 1, f"esperada 1 nota (solo el atún), got {n}"
    m_atun, m_pollo = plan["days"][0]["meals"]
    assert any("bajas en sodio" in str(s) for s in m_atun["recipe"])
    assert not any("bajas en sodio" in str(s) for s in m_pollo["recipe"])
    # idempotente
    assert go._apply_low_sodium_canned_notes(plan, _HTA) == 0


def test_lowsodium_note_scoped_to_hta_renal():
    plan = {"days": [{"meals": [{"name": "X", "ingredients": ["125g de atún en agua"], "recipe": []}]}]}
    assert go._apply_low_sodium_canned_notes(plan, {"medicalConditions": ["Diabetes tipo 2"]}) == 0, \
        "la nota es de sodio — solo HTA/ERC"
    assert go._apply_low_sodium_canned_notes(plan, _ERC) == 1


# ── P2-REVIEWER-CB-TRANSIENT ────────────────────────────────────────────────
def test_cb_open_error_is_transient_for_reviewer():
    assert go._is_reviewer_transient_error(go.LLMCircuitOpenError("Circuit Breaker OPEN para x")) is True


def test_generic_exception_still_fail_closed():
    """El whitelist sigue siendo POR TIPO: una Exception genérica (aunque diga 'Circuit Breaker')
    NO es transitoria — documenta el bug original y protege contra regresión a substring-matching."""
    assert go._is_reviewer_transient_error(Exception("Circuit Breaker OPEN para x")) is False
    assert go._is_reviewer_transient_error(ValueError("cualquier cosa")) is False


def test_no_generic_cb_raises_left():
    assert 'raise Exception(f"Circuit Breaker OPEN' not in _GRAPH, \
        "quedó un raise genérico de CB-open — usar LLMCircuitOpenError"
    assert _GRAPH.count('raise LLMCircuitOpenError(f"Circuit Breaker OPEN') >= 6, \
        "los 6 guards de invocación deben usar la excepción dedicada"


def test_marker_anchors():
    assert "P2-HTA-SALT-NORMALIZE" in _GRAPH
    assert "P2-REVIEWER-CB-TRANSIENT" in _GRAPH
    assert go.HTA_LOWSODIUM_NOTE_ENABLED is True
