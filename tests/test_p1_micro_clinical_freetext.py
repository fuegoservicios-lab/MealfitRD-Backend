"""[P1-MICRO-CLINICAL-FREETEXT · 2026-07-01] (audit micros GAP-2)

El texto libre clínico (`otherConditions`/`otherMedications` — "tengo ERC", "tomo espironolactona")
era INVISIBLE en las superficies de update: los `_micro_form` de swap-persist / regen-day /
chat-modify solo hidrataban `medicalConditions`/`medications` → el micro-closer escalaba
fibra/Mg/hierro (carga K contraindicada) y el panel recomputado perdía el techo renal K≤3000.
S1 lo cubre vía el merge P1-FORM-6; los updates no.

Fix: (a) `_enrich_clinical_from_profile` copia otherConditions/otherMedications Y pliega el
free-text en medicalConditions (merge estilo P1-FORM-6); (b) los 3 `_micro_form` incluyen las
keys y el merge.
"""
from __future__ import annotations

import re
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
_PLANS = (_BACKEND / "routers" / "plans.py").read_text(encoding="utf-8")
_TOOLS = (_BACKEND / "tools.py").read_text(encoding="utf-8")


def _fn_body(src: str, anchor: str, end_anchor: str) -> str:
    i = src.find(anchor)
    assert i != -1, f"no se encontró {anchor!r}"
    j = src.find(end_anchor, i)
    return src[i:j if j != -1 else len(src)]


def test_enrich_copies_freetext_keys():
    body = _fn_body(_PLANS, "def _enrich_clinical_from_profile", "\ndef ")
    assert '"otherConditions"' in body and '"otherMedications"' in body, \
        "_enrich_clinical_from_profile no copia otherConditions/otherMedications del perfil"
    assert "P1-MICRO-CLINICAL-FREETEXT" in body


def test_enrich_folds_freetext_into_medical_conditions():
    body = _fn_body(_PLANS, "def _enrich_clinical_from_profile", "\ndef ")
    assert 'data["medicalConditions"] = list(_mc_enr) + [str(_oc_enr).strip()]' in body, \
        "el free-text no se pliega en medicalConditions (los detectores renales solo leen esa key)"


def test_swap_persist_micro_form_includes_freetext():
    body = _fn_body(_PLANS, "def api_swap_meal_persist", "\n@router.")
    assert '"otherConditions"' in body and '"otherMedications"' in body, \
        "swap-persist: _micro_form sin otherConditions/otherMedications"
    assert "_oc_sp" in body and "_mc_sp = list(_mc_sp) + [str(_oc_sp).strip()]" in body, \
        "swap-persist: falta el merge del free-text en medicalConditions"


def test_regen_day_micro_form_includes_freetext():
    i = _PLANS.find("def _day_mutator")
    seg = _PLANS[i:i + 8000]
    assert '"otherConditions": data.get("otherConditions")' in seg, \
        "regen-day: _micro_form sin otherConditions (data viene post-enrich)"
    assert '"otherMedications": data.get("otherMedications")' in seg


def test_chat_modify_micro_form_includes_freetext():
    assert '"otherConditions": _oc_cm' in _TOOLS and '"otherMedications"' in _TOOLS, \
        "chat-modify: _micro_form_cm sin otherConditions/otherMedications"
    assert "_mc_cm = list(_mc_cm) + [str(_oc_cm).strip()]" in _TOOLS, \
        "chat-modify: falta el merge del free-text en medicalConditions"
    assert "P1-MICRO-CLINICAL-FREETEXT" in _TOOLS
