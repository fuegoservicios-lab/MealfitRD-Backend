"""[P1-UPDATE-APPETIBILITY · 2026-06-27] (audit Fase 0) Lleva a las superficies de UPDATE
(swap S3 / regenerate-day S2 / chat-modify) los detectores de APETECIBILIDAD que S1 corre en
review_plan_node: honestidad de nombre (proteína fantasma) + pareo chocante fruta+salado.

Antes: `_fix_phantom_protein_in_name` y la detección de clash SOLO corrían en S1 → un swap podía
entregar "Pollo a la Plancha" sin pollo o "Arroz con Mango".

Cubre:
  - graph_orchestrator.appetibility_fix_for_update (SSOT entrypoint para updates)
  - graph_orchestrator._meal_has_sweet_savory_clash (detector per-comida, SSOT con S1)
  - regresión: build_variety_report sigue contando el clash tras refactorizar a la SSOT
  - parser-anchored: wiring en swap (agent.py) + chat-modify (tools.py) + knob default ON
"""
from __future__ import annotations

from pathlib import Path

import pytest

import graph_orchestrator as go

_BACKEND = Path(go.__file__).resolve().parent


# ──────────────────────────── appetibility_fix_for_update ────────────────────────────

def test_phantom_protein_renamed():
    """Título lidera con proteína AUSENTE pero hay otra cárnica real → rename honesto."""
    m = {"name": "Pollo a la Plancha con Camarones",
         "ingredients": ["200g camarones", "1 taza vegetales"]}
    r = go.appetibility_fix_for_update(m)
    assert r["name_fixed"] is True
    assert "pollo" not in m["name"].lower()
    assert "camaron" in go.strip_accents(m["name"].lower())


def test_dairy_phantom_not_renamed():
    """Sin reemplazo CÁRNICO real (solo lácteo) → NO renombra (fail-safe, no estropea)."""
    m = {"name": "Cerdo a la Parrilla con Salsa de Yogurt",
         "ingredients": ["ñame", "yogurt griego", "vegetales"]}
    r = go.appetibility_fix_for_update(m)
    assert r["name_fixed"] is False
    assert m["name"] == "Cerdo a la Parrilla con Salsa de Yogurt"


def test_real_protein_not_touched():
    m = {"name": "Pechuga de Pollo al Horno", "ingredients": ["pechuga de pollo", "yuca"]}
    r = go.appetibility_fix_for_update(m)
    assert r["name_fixed"] is False
    assert m["name"] == "Pechuga de Pollo al Horno"


def test_namefix_idempotent():
    m = {"name": "Pollo a la Plancha con Camarones", "ingredients": ["camarones"]}
    go.appetibility_fix_for_update(m)
    fixed = m["name"]
    r2 = go.appetibility_fix_for_update(m)  # segunda pasada
    assert r2["name_fixed"] is False
    assert m["name"] == fixed


@pytest.mark.parametrize("name,expect", [
    ("Arroz con Mango", True),
    ("Revoltillo de Huevos con Mango", True),
    ("Coliflor salteada con Piña", True),
    ("Pollo guisado con ensalada verde", False),
    ("Yogur con Mango y Granola", False),   # fruta+lácteo NO es clash
    ("Avena con Lechosa", False),
])
def test_sweet_savory_clash_detector(name, expect):
    assert go._meal_has_sweet_savory_clash({"name": name}) is expect


def test_clash_flag_via_entrypoint():
    r = go.appetibility_fix_for_update({"name": "Arroz con Mango", "ingredients": ["arroz", "mango"]})
    assert r["sweet_savory_clash"] is True


def test_guard_off_is_noop(monkeypatch):
    monkeypatch.setattr(go, "UPDATE_APPETIBILITY_GUARD", False)
    m = {"name": "Pollo a la Plancha con Camarones", "ingredients": ["camarones"]}
    r = go.appetibility_fix_for_update(m)
    assert r == {"name_fixed": False, "sweet_savory_clash": False}
    assert m["name"] == "Pollo a la Plancha con Camarones"  # sin tocar


def test_guard_default_on():
    assert go.UPDATE_APPETIBILITY_GUARD is True


# ──────────────────────── regresión: build_variety_report SSOT ────────────────────────

def test_build_variety_report_still_counts_clash():
    """El refactor a la SSOT _meal_has_sweet_savory_clash no rompe el conteo de S1."""
    vr = go.build_variety_report({"days": [{"day": 1, "meals": [
        {"name": "Arroz con Mango", "ingredients": ["arroz", "mango"]},
        {"name": "Pollo guisado con ensalada", "ingredients": ["pollo"]},
    ]}]})
    assert vr["sweet_savory_clash"] >= 1


# ──────────────────────────── parser-anchored (no-regression) ────────────────────────────

def _src(name):
    return (_BACKEND / name).read_text(encoding="utf-8")


def test_anchor_present_in_surfaces():
    for f in ("graph_orchestrator.py", "agent.py", "tools.py"):
        assert "P1-UPDATE-APPETIBILITY" in _src(f), f"falta anchor en {f}"


def test_swap_wires_appetibility():
    src = _src("agent.py")
    assert "appetibility_fix_for_update(" in src
    assert "SWEET_SAVORY_CLASH" in src           # clash → presión de retry


def test_chat_modify_wires_appetibility():
    assert "appetibility_fix_for_update(" in _src("tools.py")
