"""[P1-MED-AMBIGUOUS-TERM-VETO · 2026-06-19] Veto de términos de medicamento ambiguos vs condiciones.

Bug (review adversaria del audit 2026-06-19): el motor de medicamentos escanea el texto médico LIBRE
(medicalConditions/otherConditions) como backstop. Detección por substring accent-free. Dos términos de
la expansión P2-MEDICATION-RULES-EXPAND eran descriptores de CONDICIÓN, no fármacos:
  - 'insulina'/'insulin' (insulin_secretagogue) matchea "resistencia a la insulina" / "resistencia
    insulinica" (condición DM2 de alta prevalencia) → advisory de HIPOGLUCEMIA erróneo + gate FS9 espurio
    a un insulino-RESISTENTE que NO toma insulina (fail-open de una gate de seguridad por un string común).
  - 'acido urico' (gout) matchea "ácido úrico alto" (condición) → prompt_block de ALOPURINOL + FS9 a
    alguien sin medicación.

Fix: (a) `term_negatives` per-regla — una cadena que matchea un term PERO contiene un negative NO activa
la regla (per-candidato); insulin_secretagogue veta 'resistencia'/'sensibilidad'/'resistente'/'insulinoma'.
(b) 'acido urico' eliminado de los terms de gout (se quedan los nombres de fármaco inequívocos). El fix es
genérico (default term_negatives=() → cero efecto en las otras reglas).
"""
from __future__ import annotations

import medication_rules as mr


def _ids(form_data):
    return {r.id for r in mr.detect_active_medications(form_data)}


# ── Falsos positivos cerrados: la CONDICIÓN no activa la regla de FÁRMACO ──
def test_insulin_resistance_condition_does_not_activate_insulin_rule():
    for cond in ("Resistencia a la insulina", "resistencia a la insulina", "Resistencia insulínica",
                 "sensibilidad a la insulina disminuida"):
        assert "insulin_secretagogue" not in _ids({"medicalConditions": [cond]}), cond
        assert "insulin_secretagogue" not in _ids({"otherConditions": cond}), cond


def test_uric_acid_condition_does_not_activate_gout_rule():
    for cond in ("Ácido úrico alto", "acido urico alto", "Hiperuricemia"):
        assert "gout" not in _ids({"medicalConditions": [cond]}), cond


# ── No regresión: el FÁRMACO real sigue detectándose ──
def test_real_insulin_med_still_detected():
    assert "insulin_secretagogue" in _ids({"medications": ["Insulina"]})
    assert "insulin_secretagogue" in _ids({"medications": ["Glibenclamida"]})
    # Texto libre legítimo (sin 'resistencia') sigue contando por el backstop.
    assert "insulin_secretagogue" in _ids({"otherConditions": "me inyecto insulina en la noche"})


def test_real_gout_med_still_detected():
    assert "gout" in _ids({"medications": ["Alopurinol"]})
    assert "gout" in _ids({"medications": ["Febuxostat"]})


# ── Co-ocurrencia: fármaco real + condición resistente → la regla SÍ activa (gana el fármaco real) ──
def test_real_insulin_plus_resistance_condition_still_active():
    ids = _ids({"medications": ["Insulina"], "medicalConditions": ["Resistencia a la insulina"]})
    assert "insulin_secretagogue" in ids


# ── El veto es opt-in: las demás reglas (default term_negatives=()) no se ven afectadas ──
def test_veto_default_empty_is_noop_for_other_rules():
    assert mr._RULES_BY_ID["ace_arb"].term_negatives == ()
    assert "ace_arb" in _ids({"medications": ["Losartán"]})
    assert "metformin" in _ids({"medications": ["Metformina"]})
