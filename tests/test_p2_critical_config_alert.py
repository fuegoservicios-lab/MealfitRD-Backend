"""[P2-CRITICAL-CONFIG-ALERT · 2026-06-15] Alerta de configuración crítica mal seteada en prod (gap-audit G7+G10).

G7: el motor de precisión (MACRO_SOLVER_ENABLED) tiene default de código False y solo está ON por el .env del
VPS → un redeploy con env limpio lo apaga SIN aviso (proteína ~16% MAPE). G10: los guards de seguridad
clínica (cap renal/alérgenos/gate de revisión/reglas por condición) default True → un override a False los
apaga sin alerta runtime. `get_critical_config_warnings()` (pura) los detecta SOLO en producción; el lifespan
los emite/resuelve en system_alerts.

También ancla el decouple del cap renal (G10): Guard 1 gateado SOLO por RENAL_CAP_ENABLED, no por
CONDITION_RULES_ENABLED — apagar las reglas-por-condición (calidad) NO debe apagar el cap renal (seguridad).

Tests funcionales (monkeypatch de globals + is_production) + parser-based del decouple.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

import knobs
import graph_orchestrator as go

_SRC = (Path(__file__).resolve().parent.parent / "graph_orchestrator.py").read_text(encoding="utf-8")

_SAFETY_ATTRS = ("RENAL_CAP_ENABLED", "ALLERGEN_HARD_GUARD", "ALLERGEN_SUBSTITUTION_ENABLED",
                 "PRO_REVIEW_FLAG_ENABLED", "CONDITION_RULES_ENABLED", "PROTEIN_FLOOR_HARD_GATE")


def _all_healthy(monkeypatch):
    """Producción + todo bien (solver ON, guards ON)."""
    monkeypatch.setattr(knobs, "is_production", lambda: True)
    monkeypatch.setattr(go, "MACRO_SOLVER_ENABLED", True)
    for a in _SAFETY_ATTRS:
        monkeypatch.setattr(go, a, True)


def test_no_warnings_outside_production(monkeypatch):
    monkeypatch.setattr(knobs, "is_production", lambda: False)
    monkeypatch.setattr(go, "MACRO_SOLVER_ENABLED", False)  # apagado, pero no-prod → sin alerta
    for a in _SAFETY_ATTRS:
        monkeypatch.setattr(go, a, False)
    assert go.get_critical_config_warnings() == []


def test_no_warnings_when_all_healthy(monkeypatch):
    _all_healthy(monkeypatch)
    assert go.get_critical_config_warnings() == []


def test_macro_engine_off_in_prod_warns(monkeypatch):
    _all_healthy(monkeypatch)
    monkeypatch.setattr(go, "MACRO_SOLVER_ENABLED", False)
    warns = go.get_critical_config_warnings()
    keys = {w["alert_key"] for w in warns}
    assert "macro_engine_disabled_in_prod" in keys
    assert all(w["severity"] == "high" for w in warns)


def test_safety_guard_off_in_prod_warns(monkeypatch):
    _all_healthy(monkeypatch)
    monkeypatch.setattr(go, "RENAL_CAP_ENABLED", False)
    keys = {w["alert_key"] for w in go.get_critical_config_warnings()}
    assert "safety_guard_disabled_in_prod:MEALFIT_RENAL_CAP" in keys


def test_each_safety_knob_has_its_own_alert(monkeypatch):
    """Cada knob de seguridad apagado emite SU propia alerta (no se confunden)."""
    _env_to_attr = {
        "MEALFIT_RENAL_CAP": "RENAL_CAP_ENABLED",
        "MEALFIT_ALLERGEN_HARD_GUARD": "ALLERGEN_HARD_GUARD",
        "MEALFIT_ALLERGEN_SUBSTITUTION": "ALLERGEN_SUBSTITUTION_ENABLED",
        "MEALFIT_PRO_REVIEW_FLAG": "PRO_REVIEW_FLAG_ENABLED",
        "MEALFIT_CONDITION_RULES": "CONDITION_RULES_ENABLED",
        "MEALFIT_PROTEIN_FLOOR_HARD_GATE": "PROTEIN_FLOOR_HARD_GATE",
    }
    for env_name, attr in _env_to_attr.items():
        _all_healthy(monkeypatch)
        monkeypatch.setattr(go, attr, False)
        keys = {w["alert_key"] for w in go.get_critical_config_warnings()}
        assert f"safety_guard_disabled_in_prod:{env_name}" in keys, f"falta alerta para {env_name}"


def test_renal_cap_guard1_decoupled_from_condition_rules():
    """Parser-based (G10): el Guard 1 renal de la capa clínica se gatea SOLO por RENAL_CAP_ENABLED,
    NO por `CONDITION_RULES_ENABLED and RENAL_CAP_ENABLED` (apagar calidad ≠ apagar seguridad)."""
    start = _SRC.find("def _apply_deterministic_clinical_layer")
    body = _SRC[start: start + 40000]
    # El gate de Guard 1 (con renal_protein_cap.applied) ya NO debe llevar CONDITION_RULES_ENABLED.
    assert "P2-SAFETY-KNOB-DECOUPLE" in body, "falta el anchor del decouple del cap renal."
    assert not re.search(r"if\s*\(\s*CONDITION_RULES_ENABLED\s+and\s+RENAL_CAP_ENABLED\s+and\s+_db", body), (
        "El Guard 1 renal NO debe gatearse por `CONDITION_RULES_ENABLED and RENAL_CAP_ENABLED` (G10)."
    )
    assert re.search(r"if\s*\(\s*RENAL_CAP_ENABLED\s+and\s+_db\s+is\s+not\s+None", body), (
        "El Guard 1 renal debe gatearse SOLO por `RENAL_CAP_ENABLED and _db is not None`."
    )
