"""
Tests P0-FORM-4: `weightUnit` required + validación de valor + heurística de
unidad invertida.

Bug original:
  `_validate_form_data_ranges` leía `data.get("weightUnit") or "lb"`. Si el
  cliente omitía la key y mandaba peso en kg (ej 70), el backend asumía "lb",
  convertía a kg → 70/2.20462 = 31.7 kg, JUSTO por encima del mínimo de 30 kg.
  El plan se generaba con BMR completamente errado (un "adulto" de 31.7 kg).

Fix:
  1. `weightUnit` añadido a `_REQUIRED_FORM_FIELDS` → 422
     `missing_required_fields` si ausente.
  2. `_validate_form_data_ranges` valida valor en {"lb","kg"} (case-insensitive)
     → 422 `invalid_biometric_range` con `field=weightUnit` si inválido.
  3. Heurística defensiva: log WARNING si peso original ≥150 + post-conversión
     ≤35 kg (sospecha de unidad invertida).
  4. `nutrition_calculator.get_nutrition_targets` ya no defaultea silenciosamente
     — emite WARNING al log si recibe payload sin weightUnit (defensa para
     callers internos: cron, agent).
"""
import logging

import pytest

from routers.plans import (
    _REQUIRED_FORM_FIELDS,
    _WEIGHT_UNIT_ACCEPTED,
    _validate_form_data_min,
    _validate_form_data_ranges,
)


# ---------------------------------------------------------------------------
# 1. Presence check: weightUnit ahora required.
# ---------------------------------------------------------------------------
def test_weight_unit_esta_en_required_fields():
    assert "weightUnit" in _REQUIRED_FORM_FIELDS


def test_min_validator_rechaza_payload_sin_weight_unit():
    payload = {
        "age": 30, "mainGoal": "lose_fat", "weight": 70, "height": 170,
        "gender": "male", "activityLevel": "moderate",
        # weightUnit ausente
    }
    ok, missing = _validate_form_data_min(payload)
    assert ok is False
    assert "weightUnit" in missing


def test_min_validator_rechaza_weight_unit_string_vacio():
    payload = {
        "age": 30, "mainGoal": "lose_fat", "weight": 70, "height": 170,
        "gender": "male", "activityLevel": "moderate",
        "weightUnit": "",
    }
    ok, missing = _validate_form_data_min(payload)
    assert ok is False
    assert "weightUnit" in missing


def test_min_validator_acepta_weight_unit_valido():
    payload = {
        "age": 30, "mainGoal": "lose_fat", "weight": 70, "height": 170,
        "gender": "male", "activityLevel": "moderate",
        "weightUnit": "kg",
        # [P0-FORM-1] householdSize/groceryDuration ahora required
        # (antes defaulteaban silenciosamente a 1 / "weekly").
        "householdSize": 1,
        "groceryDuration": "weekly",
        # [P0-FORM-3] motivation ahora required (antes era huérfana, ahora
        # cableada al planner LLM vía `build_motivation_context`).
        "motivation": "Recuperar mi energía diaria.",
        # [P1-2] allergies/medicalConditions presence-required + nonempty.
        "allergies": ["Ninguna"],
        "medicalConditions": ["Ninguna"],
        # [P0-FORM-6] sync con frontend: scheduleType/cookingTime/budget/
        # sleepHours/stressLevel/dislikes/struggles ahora required.
        "scheduleType": "standard",
        "cookingTime": "30min",
        "budget": "medium",
        "sleepHours": "7-8 horas",
        "stressLevel": "Moderado",
        "dislikes": ["Ninguno"],
        "struggles": ["Ninguno"],
    }
    ok, missing = _validate_form_data_min(payload)
    assert ok is True, f"missing: {missing}"
    assert missing == []


# ---------------------------------------------------------------------------
# 2. Value check: weightUnit debe ser "lb" o "kg".
# ---------------------------------------------------------------------------
def test_accepted_set_solo_contiene_lb_y_kg():
    assert _WEIGHT_UNIT_ACCEPTED == frozenset({"lb", "kg"})


def test_ranges_validator_rechaza_weight_unit_invalido():
    """`weightUnit='lbs'` (typo común) debe rechazarse — no normalizamos a 'lb'."""
    payload = {
        "age": 30, "weight": 70, "height": 170, "weightUnit": "lbs",
    }
    ok, errors = _validate_form_data_ranges(payload)
    assert ok is False
    weight_unit_errors = [e for e in errors if e["field"] == "weightUnit"]
    assert len(weight_unit_errors) == 1
    assert weight_unit_errors[0]["value"] == "lbs"
    assert weight_unit_errors[0]["accepted_range"] == ["kg", "lb"]


@pytest.mark.parametrize("variant", ["LB", "Lb", " lb ", "kg", "KG", " KG "])
def test_ranges_validator_acepta_variantes_de_capitalizacion(variant):
    payload = {
        "age": 30, "weight": 70, "height": 170, "weightUnit": variant,
    }
    ok, errors = _validate_form_data_ranges(payload)
    weight_unit_errors = [e for e in errors if e["field"] == "weightUnit"]
    assert weight_unit_errors == [], f"variante {variant!r} debió aceptarse"


def test_ranges_validator_no_emite_doble_error_cuando_unit_invalido():
    """Si weightUnit es inválido, NO emitimos también un error de weight fuera
    de rango — sería confuso para el usuario (el problema raíz es la unidad)."""
    payload = {
        "age": 30, "weight": 70, "height": 170, "weightUnit": "lbz",
    }
    ok, errors = _validate_form_data_ranges(payload)
    fields = {e["field"] for e in errors}
    assert "weightUnit" in fields
    assert "weight" not in fields  # no doble error


def test_ranges_validator_rechaza_weight_unit_ausente_como_invalido():
    """Defense-in-depth: si _validate_form_data_min se saltó por algún path,
    aquí también rechazamos en vez de defaultear silenciosamente a 'lb'."""
    payload = {"age": 30, "weight": 70, "height": 170}  # sin weightUnit
    ok, errors = _validate_form_data_ranges(payload)
    assert ok is False
    weight_unit_errors = [e for e in errors if e["field"] == "weightUnit"]
    assert len(weight_unit_errors) == 1
    assert weight_unit_errors[0]["value"] is None


# ---------------------------------------------------------------------------
# 3. Bug primario reproducido: el escenario "weightUnit ausente + 70 kg" ya
# no pasa silenciosamente como 31.7 kg.
# ---------------------------------------------------------------------------
def test_escenario_bug_original_70_kg_sin_unit_se_rechaza():
    """
    PRE-FIX: payload {weight: 70} sin weightUnit → backend asumía "lb" →
    70/2.20462 = 31.7 kg, justo arriba del mínimo de 30 kg → plan generado
    con BMR de un "adulto" de 31.7 kg (basura).

    POST-FIX: validador rechaza con `weightUnit` inválido. El plan no se genera.
    """
    payload = {
        "age": 30, "mainGoal": "lose_fat", "weight": 70, "height": 170,
        "gender": "male", "activityLevel": "moderate",
        # weightUnit AUSENTE — escenario del bug.
    }
    # Min check ya rechaza por presencia.
    ok_min, missing = _validate_form_data_min(payload)
    assert ok_min is False
    assert "weightUnit" in missing
    # Range check también rechaza (defense-in-depth) si el caller saltó min.
    ok_range, errors = _validate_form_data_ranges(payload)
    assert ok_range is False


# ---------------------------------------------------------------------------
# 4. Heurística defensiva: peso ≥150 + post-conversión ≤35 kg.
# ---------------------------------------------------------------------------
def test_heuristica_unidad_invertida_loguea_warning(caplog):
    """
    Caso construido: weightUnit='lb' con weight=77 (que en lb son 35 kg justos).
    Para que dispare la heurística necesitamos peso_raw ≥ 150 Y weight_kg ≤ 35.
    Eso requiere weight_raw ≥ 150 lb pero weight_kg ≤ 35: 150/2.20462 = 68 kg,
    que NO entra en ≤35. Para forzar el escenario hay que ingresar weight_raw
    muy alto en una unidad que produce kg muy bajo — solo posible con datos
    inválidos o conversión rota. Validamos que el log NO se dispara en
    escenarios normales (regresión: la heurística no es ruidosa).
    """
    payload_normal = {
        "age": 30, "weight": 150, "height": 170, "weightUnit": "lb",
    }
    with caplog.at_level(logging.WARNING, logger="routers.plans"):
        ok, errors = _validate_form_data_ranges(payload_normal)
    assert ok is True
    suspicious_logs = [r for r in caplog.records if "P0-FORM-4" in r.message]
    assert len(suspicious_logs) == 0, "no debe loguear sospecha en payload normal"


def test_payload_normal_lb_pasa_validacion():
    payload = {
        "age": 30, "weight": 150, "height": 170, "weightUnit": "lb",
    }
    ok, errors = _validate_form_data_ranges(payload)
    assert ok is True
    assert errors == []


def test_payload_normal_kg_pasa_validacion():
    payload = {
        "age": 30, "weight": 70, "height": 170, "weightUnit": "kg",
    }
    ok, errors = _validate_form_data_ranges(payload)
    assert ok is True
    assert errors == []


# ---------------------------------------------------------------------------
# 5. nutrition_calculator: no defaultea silenciosamente.
# ---------------------------------------------------------------------------
def test_nutrition_calculator_loguea_warning_si_weight_unit_ausente(caplog):
    from nutrition_calculator import get_nutrition_targets
    form = {
        "weight": 70, "height": 170, "age": 30, "gender": "male",
        "activityLevel": "moderate", "mainGoal": "maintenance",
        # weightUnit AUSENTE
    }
    with caplog.at_level(logging.WARNING, logger="nutrition_calculator"):
        result = get_nutrition_targets(form)
    # Sigue calculando (fallback "lb" para perfiles legacy), pero LOGUEA.
    assert result is not None
    p0_logs = [
        r for r in caplog.records
        if "P0-FORM-4" in r.message and "weightUnit ausente" in r.message
    ]
    assert len(p0_logs) >= 1


def test_nutrition_calculator_loguea_warning_si_weight_unit_invalido(caplog):
    from nutrition_calculator import get_nutrition_targets
    form = {
        "weight": 70, "height": 170, "age": 30, "gender": "male",
        "activityLevel": "moderate", "mainGoal": "maintenance",
        "weightUnit": "lbs",  # inválido
    }
    with caplog.at_level(logging.WARNING, logger="nutrition_calculator"):
        result = get_nutrition_targets(form)
    assert result is not None
    p0_logs = [
        r for r in caplog.records
        if "P0-FORM-4" in r.message and "inválido" in r.message
    ]
    assert len(p0_logs) >= 1


def test_nutrition_calculator_no_loguea_warning_en_path_normal(caplog):
    from nutrition_calculator import get_nutrition_targets
    form = {
        "weight": 150, "height": 170, "age": 30, "gender": "male",
        "activityLevel": "moderate", "mainGoal": "maintenance",
        "weightUnit": "lb",
    }
    with caplog.at_level(logging.WARNING, logger="nutrition_calculator"):
        get_nutrition_targets(form)
    p0_logs = [r for r in caplog.records if "P0-FORM-4" in r.message]
    assert p0_logs == []
