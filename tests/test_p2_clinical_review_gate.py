"""[P2-CLINICAL-REVIEW-GATE · 2026-07-03] Flags clínicos del panel opt-in vetan
el bypass del reviewer médico.

Hallazgo del smoke E2E 2026-07-03: un perfil con HbA1c prediabética en
`clinical_profile` pero SIN condiciones declaradas recibía la influencia
clínica en la GENERACIÓN, pero el reviewer LLM se bypasseaba ("Sin
restricciones declaradas") → la nota clínica del reviewer nunca se ejercitaba.

Contrato:
  - `clinical_profile_active_flags` (prompts/plan_generator.py) es el SSOT de
    umbrales: labs fuera de rango + pérdida no intencional → flag. GI y
    entrenamiento NO generan flag (guía blanda).
  - El builder de contexto consume los flags (paridad texto↔flag — anti-drift).
  - `review_plan_node`: bypass SOLO si además `not _clinical_flags`.
  - Knob `MEALFIT_CLINICAL_FLAGS_FORCE_REVIEW` (default True) restaura el
    bypass legacy sin redeploy.
"""
from __future__ import annotations

import re
from pathlib import Path

from prompts.plan_generator import (
    build_clinical_profile_context,
    clinical_profile_active_flags,
)

_ORCH_SRC = (Path(__file__).resolve().parent.parent / "graph_orchestrator.py").read_text(encoding="utf-8")


def _fd(cp):
    return {"clinical_profile": cp}


# ---------------------------------------------------------------------------
# 1. Helper de flags — SSOT de umbrales
# ---------------------------------------------------------------------------
def test_no_flags_without_data():
    assert clinical_profile_active_flags(None) == []
    assert clinical_profile_active_flags({}) == []
    assert clinical_profile_active_flags(_fd({})) == []
    assert clinical_profile_active_flags(_fd({"labs": {}})) == []


def test_gi_and_training_never_flag():
    assert clinical_profile_active_flags(_fd({
        "giSymptoms": ["reflujo", "estrenimiento"],
        "training": {"type": "fuerza", "timeOfDay": "noche", "daysPerWeek": 5},
    })) == []


def test_glucemia_thresholds():
    assert "glucemia_rango_diabetes" in clinical_profile_active_flags(_fd({"labs": {"hba1c": 6.5}}))
    assert "glucemia_rango_diabetes" in clinical_profile_active_flags(_fd({"labs": {"glucosa_ayunas": 126}}))
    flags = clinical_profile_active_flags(_fd({"labs": {"hba1c": 6.0}}))
    assert flags == ["glucemia_prediabetes"]
    # Normal → sin flag.
    assert clinical_profile_active_flags(_fd({"labs": {"hba1c": 5.2, "glucosa_ayunas": 88}})) == []
    # Diabetes y prediabetes son excluyentes (elif).
    both = clinical_profile_active_flags(_fd({"labs": {"hba1c": 7.0, "glucosa_ayunas": 110}}))
    assert "glucemia_rango_diabetes" in both and "glucemia_prediabetes" not in both


def test_remaining_lab_thresholds():
    assert "lipidos_elevados" in clinical_profile_active_flags(_fd({"labs": {"ldl": 160}}))
    assert "tfg_reducido" in clinical_profile_active_flags(_fd({"labs": {"tfg": 59}}))
    assert clinical_profile_active_flags(_fd({"labs": {"tfg": 60}})) == []
    assert "acido_urico_elevado" in clinical_profile_active_flags(_fd({"labs": {"acido_urico": 7}}))
    assert "hemoglobina_baja" in clinical_profile_active_flags(_fd({"labs": {"hemoglobina": 11.5}}))
    assert "vitamina_d_baja" in clinical_profile_active_flags(_fd({"labs": {"vitamina_d": 15}}))
    assert "tsh_elevada" in clinical_profile_active_flags(_fd({"labs": {"tsh": 5.0}}))


def test_unintentional_loss_flag_and_nested_transport():
    assert "perdida_no_intencional" in clinical_profile_active_flags(
        _fd({"weightHistory": {"unintentionalLoss": True}})
    )
    # Transporte anidado en health_profile (mismo contrato que el builder).
    assert "tfg_reducido" in clinical_profile_active_flags(
        {"health_profile": {"clinical_profile": {"labs": {"tfg": 40}}}}
    )


# ---------------------------------------------------------------------------
# 2. Anti-drift: flag activo ⟺ el builder emite el texto correspondiente
# ---------------------------------------------------------------------------
def test_flag_text_parity():
    # Markers = frases EXCLUSIVAS del texto de alarma de cada flag (no del
    # listado de valores — "ácido úrico 5 mg/dL" aparece aunque sea normal).
    cases = [
        ({"labs": {"hba1c": 7.0}}, "COMPATIBLE CON DIABETES"),
        ({"labs": {"hba1c": 6.0}}, "PREDIABETES"),
        ({"labs": {"ldl": 170}}, "lipídico elevado"),
        ({"labs": {"tfg": 45}}, "nefrólogo"),
        ({"labs": {"acido_urico": 8}}, "limita vísceras"),
        ({"weightHistory": {"unintentionalLoss": True}}, "NO INTENCIONAL"),
    ]
    for cp, marker in cases:
        flags = clinical_profile_active_flags(_fd(cp))
        text = build_clinical_profile_context(_fd(cp))
        assert flags, f"esperaba flag para {cp}"
        assert marker in text, f"flag activo {flags} sin texto '{marker}'"
    # Inverso: labs normales → cero flags y cero textos de alarma.
    normal = {"labs": {"hba1c": 5.2, "ldl": 90, "tfg": 95, "acido_urico": 5}}
    assert clinical_profile_active_flags(_fd(normal)) == []
    text = build_clinical_profile_context(_fd(normal))
    for marker in ("PREDIABETES", "COMPATIBLE CON DIABETES", "nefrólogo", "limita vísceras", "NO INTENCIONAL"):
        assert marker not in text


# ---------------------------------------------------------------------------
# 3. Gate del reviewer (parser-based sobre graph_orchestrator)
# ---------------------------------------------------------------------------
def test_bypass_condition_includes_clinical_flags():
    m = re.search(
        r"if not _has_real_medical_flags\(allergies\).*?and not _clinical_flags:",
        _ORCH_SRC, re.DOTALL,
    )
    assert m, "el bypass del reviewer debe incluir `and not _clinical_flags`"
    assert "Bypassing LLM Reviewer" in _ORCH_SRC[m.end():m.end() + 300]


def test_flags_computed_under_knob():
    # El cómputo de flags vive detrás del knob (rollback sin redeploy).
    m = re.search(
        r"if CLINICAL_FLAGS_FORCE_REVIEW:\s*\n\s*try:\s*\n\s*_clinical_flags = clinical_profile_active_flags\(form_data\)",
        _ORCH_SRC,
    )
    assert m, "clinical_profile_active_flags debe computarse solo con el knob ON"


def test_knob_registered():
    # El knob se auto-registra al importar graph_orchestrator (donde vive el
    # _env_bool) — importar knobs solo no basta.
    import graph_orchestrator  # noqa: F401
    from knobs import get_knobs_registry_snapshot
    assert "MEALFIT_CLINICAL_FLAGS_FORCE_REVIEW" in get_knobs_registry_snapshot()


def test_import_present():
    assert "clinical_profile_active_flags," in _ORCH_SRC
