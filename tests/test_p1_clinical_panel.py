"""[P1-CLINICAL-PANEL · 2026-07-03] Perfil Clínico Avanzado opt-in.

Cubre las 4 piezas:
  1. `build_clinical_profile_context` (prompts/plan_generator.py) — bloque de
     prompt con flags HONESTOS de labs (guía, nunca diagnóstico), historia
     ponderal (pérdida no intencional = red flag), directivas GI y timing de
     entrenamiento. "" si no hay datos accionables (no-op → prompt-cache).
  2. `_clean_clinical_profile` (routers/user_data.py) — validador del endpoint
     (rangos anti-typo de labs, enums GI/entrenamiento, sentinel 'ninguno'
     exclusivo, freeText capado).
  3. Inyección al pipeline: `clinical_directives += build_clinical_profile_context`
     (planner + day-gen) y `_clinical_panel_note` en el prompt del revisor médico.
  4. Endpoints GET/PUT `/user/preferences/clinical-profile` presentes con auth.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from prompts.plan_generator import build_clinical_profile_context

_BACKEND = Path(__file__).resolve().parent.parent
_ORCH_SRC = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
_UD_SRC = (_BACKEND / "routers" / "user_data.py").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Builder — no-op y transportes
# ---------------------------------------------------------------------------
def test_builder_noop_without_data():
    assert build_clinical_profile_context(None) == ""
    assert build_clinical_profile_context({}) == ""
    assert build_clinical_profile_context({"clinical_profile": None}) == ""
    assert build_clinical_profile_context({"clinical_profile": {}}) == ""
    # Secciones presentes pero vacías → sigue siendo no-op.
    assert build_clinical_profile_context(
        {"clinical_profile": {"labs": {}, "giSymptoms": [], "training": {"type": "", "timeOfDay": "", "daysPerWeek": 0}, "freeText": ""}}
    ) == ""


def test_builder_nested_health_profile_transport():
    out = build_clinical_profile_context(
        {"health_profile": {"clinical_profile": {"giSymptoms": ["reflujo"]}}}
    )
    assert "Reflujo" in out


# ---------------------------------------------------------------------------
# 2. Builder — flags de labs honestos (guía + confirmación profesional)
# ---------------------------------------------------------------------------
def test_hba1c_diabetic_range_flag():
    out = build_clinical_profile_context({"clinical_profile": {"labs": {"hba1c": 7.2}}})
    assert "COMPATIBLE CON DIABETES" in out
    assert "confirmación" in out.lower() or "profesional" in out.lower()


def test_hba1c_prediabetes_flag():
    out = build_clinical_profile_context({"clinical_profile": {"labs": {"hba1c": 6.0}}})
    assert "PREDIABETES" in out
    assert "COMPATIBLE CON DIABETES" not in out


def test_tfg_low_renal_moderation():
    out = build_clinical_profile_context({"clinical_profile": {"labs": {"tfg": 45}}})
    assert "renal" in out.lower()
    assert "proteína" in out.lower()
    assert "REQUIERE" in out


def test_lipids_and_uric_flags():
    out = build_clinical_profile_context(
        {"clinical_profile": {"labs": {"ldl": 170, "acido_urico": 8.1}}}
    )
    assert "lipídico" in out.lower()
    assert "úrico" in out.lower()


def test_normal_labs_no_flags():
    out = build_clinical_profile_context(
        {"clinical_profile": {"labs": {"hba1c": 5.2, "ldl": 90, "tfg": 95}}}
    )
    # Lista los valores pero SIN flags de alarma.
    assert "HbA1c" in out
    assert "PREDIABETES" not in out and "COMPATIBLE CON DIABETES" not in out
    assert "renal" not in out.lower()


# ---------------------------------------------------------------------------
# 3. Builder — ponderal, GI, entrenamiento
# ---------------------------------------------------------------------------
def test_unintentional_loss_red_flag():
    out = build_clinical_profile_context(
        {"clinical_profile": {"weightHistory": {"unintentionalLoss": True}}}
    )
    assert "NO INTENCIONAL" in out
    assert "NO apliques déficit" in out


def test_gi_directives():
    out = build_clinical_profile_context(
        {"clinical_profile": {"giSymptoms": ["reflujo", "estrenimiento"]}}
    )
    assert "cena LIGERA" in out
    assert "fibra" in out.lower()


def test_training_timing_block():
    out = build_clinical_profile_context(
        {"clinical_profile": {"training": {"type": "fuerza", "timeOfDay": "noche", "daysPerWeek": 4}}}
    )
    assert "fuerza" in out
    assert "4x/semana" in out
    assert "proteína" in out.lower()


# ---------------------------------------------------------------------------
# 4. Inyección al pipeline (parser-based)
# ---------------------------------------------------------------------------
def test_injected_into_clinical_directives():
    # Composición: viaja con las directivas clínicas → planner + day-gen.
    assert re.search(
        r"clinical_directives\s*\+=\s*build_clinical_profile_context\(form_data\)",
        _ORCH_SRC,
    ), "build_clinical_profile_context debe sumarse a clinical_directives"


def test_injected_into_medical_reviewer_prompt():
    assert "_clinical_panel_note = build_clinical_profile_context(form_data)" in _ORCH_SRC
    assert "{_baria_note}{_clinical_panel_note}" in _ORCH_SRC, (
        "el prompt del revisor médico debe interpolar _clinical_panel_note"
    )


# ---------------------------------------------------------------------------
# 5. Endpoints + validador
# ---------------------------------------------------------------------------
def test_endpoints_present_with_auth():
    assert '@router.get("/user/preferences/clinical-profile")' in _UD_SRC
    assert '@router.put("/user/preferences/clinical-profile")' in _UD_SRC
    # Ambos endpoints dependen del user autenticado (I2).
    _block = _UD_SRC.split('clinical-profile")')[1]
    assert "get_verified_user_id" in _block


def _import_cleaner():
    try:
        from routers.user_data import _clean_clinical_profile
        return _clean_clinical_profile
    except Exception as e:  # pragma: no cover - import-heavy en algunos entornos
        pytest.skip(f"routers.user_data no importable en este entorno: {e}")


def test_cleaner_valid_payload_normalizes():
    clean = _import_cleaner()
    out = clean({
        "labs": {"hba1c": "6,1", "tfg": 88, "glucosa_ayunas": ""},
        "weightHistory": {"unit": "lb", "maxWeight": "230", "unintentionalLoss": False},
        "giSymptoms": ["Reflujo"],
        "training": {"type": "mixto", "timeOfDay": "manana", "daysPerWeek": "3"},
        "freeText": "  Me operaron de la vesícula.  ",
    })
    assert out["labs"]["hba1c"] == 6.1            # coma decimal normalizada
    assert "glucosa_ayunas" not in out["labs"]     # vacío → omitido
    assert out["weightHistory"]["maxWeight"] == 230.0
    assert out["giSymptoms"] == ["reflujo"]
    assert out["training"]["daysPerWeek"] == 3
    assert out["freeText"] == "Me operaron de la vesícula."


def test_cleaner_rejects_out_of_range_lab():
    clean = _import_cleaner()
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc:
        clean({"labs": {"hba1c": 55}})
    assert exc.value.status_code == 422


def test_cleaner_rejects_bad_enums():
    clean = _import_cleaner()
    from fastapi import HTTPException
    with pytest.raises(HTTPException):
        clean({"giSymptoms": ["dolor de cabeza"]})
    with pytest.raises(HTTPException):
        clean({"training": {"type": "yoga aereo"}})
    with pytest.raises(HTTPException):
        clean({"training": {"type": "", "timeOfDay": "", "daysPerWeek": 9}})


def test_cleaner_ninguno_exclusive_and_unit_required():
    clean = _import_cleaner()
    out = clean({"giSymptoms": ["ninguno", "reflujo"]})
    assert out["giSymptoms"] == ["reflujo"]        # sentinel exclusivo
    from fastapi import HTTPException
    with pytest.raises(HTTPException):
        clean({"weightHistory": {"maxWeight": 230}})  # pesos sin unit → 422
