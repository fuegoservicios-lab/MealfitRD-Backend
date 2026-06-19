"""[P1-PREGNANCY-INTAKE-CAPTURE · 2026-06-19] (audit fresco P1-1) Captura explícita de embarazo/lactancia.

Bug (audit 2026-06-19): el gate de seguridad P1-PREGNANCY-DEFICIT-GATE (bloquea el déficit calórico en
embarazo, nutrition_calculator.py) está vivo, pero su disparador dependía 100% de que la usuaria ESCRIBIERA
"embarazo" en el texto libre — el chip-set de QMedical no ofrecía embarazo/lactancia. Punto ciego de alto
riesgo/prevalencia (daño fetal/de lactancia por déficit), la simétrica faltante del campo `medications`.

Fix (solo-frontend, cero cambio backend): chips gender-gated "Embarazo"/"Lactancia" en QMedical que escriben
a `medicalConditions` con labels que matchean PREGNANCY_CONDITION_TERMS → disparan el gate de déficit, la
ConditionRule de embarazo (folato/hierro/listeria) y el reviewer médico/FS9. Este test ancla (1) los chips +
el gate de género en el source del wizard, (2) que los labels matchean los términos backend, (3) que el
backend dispara el gate y la ConditionRule al recibir esos valores en `medicalConditions`.
"""
from __future__ import annotations

from pathlib import Path

from constants import PREGNANCY_CONDITION_TERMS, strip_accents
import condition_rules as cr
import nutrition_calculator as nc


_IQ_PATH = (Path(__file__).resolve().parents[2] / "frontend" / "src" / "components" / "assessment"
            / "questions" / "InteractiveQuestions.jsx")

# Labels exactos de los chips del wizard (deben matchear PREGNANCY_CONDITION_TERMS).
_CHIP_LABELS = ("Embarazo", "Lactancia")


# ── A. Parser-anchor del wizard (rename → falla el test antes de romper prod) ──
def test_wizard_has_pregnancy_chips_gender_gated():
    src = _IQ_PATH.read_text(encoding="utf-8")
    assert "P1-PREGNANCY-INTAKE-CAPTURE" in src
    assert "['Embarazo', 'Lactancia']" in src, "los chips de embarazo/lactancia deben existir (SSOT)"
    assert "PREGNANCY_CHIP_LABELS" in src, "los chips deben leer del SSOT compartido"
    assert "formData.gender === 'female'" in src, "los chips deben estar gateados a mujeres"


def test_wizard_cleans_pregnancy_orphan_on_gender_change():
    # Si el usuario marca embarazo y luego (back-nav) cambia el género a hombre, el chip se oculta pero el
    # valor seguía vivo en medicalConditions → override silencioso. QGender lo limpia al fijar un género
    # no-mujer. Ancla el cleanup en el source del wizard.
    src = _IQ_PATH.read_text(encoding="utf-8")
    assert "value !== 'female'" in src, "QGender debe limpiar embarazo cuando el género no es mujer"
    assert "PREGNANCY_CHIP_LABELS.includes(c)" in src, "el cleanup debe filtrar por el SSOT de labels"


# ── B. Los labels de los chips disparan los términos backend ──
def test_chip_labels_match_pregnancy_terms():
    for label in _CHIP_LABELS:
        norm = strip_accents(label.lower())
        assert any(t in norm for t in PREGNANCY_CONDITION_TERMS), \
            f"el chip {label!r} debe matchear algún PREGNANCY_CONDITION_TERM"


# ── C. El backend dispara el gate + la ConditionRule al recibir los chips en medicalConditions ──
def test_pregnancy_chip_triggers_deficit_gate():
    for label in _CHIP_LABELS:
        assert nc._is_pregnancy_or_lactation({"medicalConditions": [label]}) is True, label


def test_pregnancy_chip_triggers_condition_rule():
    for label in _CHIP_LABELS:
        ids = {r.id for r in cr.detect_active_rules({"medicalConditions": [label]})}
        assert "pregnancy" in ids, label


def test_non_pregnancy_condition_does_not_trigger():
    assert nc._is_pregnancy_or_lactation({"medicalConditions": ["Hipertensión"]}) is False
    ids = {r.id for r in cr.detect_active_rules({"medicalConditions": ["Hipertensión"]})}
    assert "pregnancy" not in ids
