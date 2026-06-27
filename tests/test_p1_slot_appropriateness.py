"""[P1-SLOT-APPROPRIATENESS · 2026-06-27] (audit G4) Coherencia comida↔HORARIO es-DO.

El ejemplo central del owner: "arroz de noche" no se come; el arroz/locrio/pasta van en
almuerzo/cena, NUNCA en desayuno; la cena es ligera. Antes esto vivía SOLO en el prompt
(advisory, no garantizado, skip en intento>1) y NO existía en S2/S3. Este P-fix lo vuelve un
GATE determinista en S1 (review_plan_node) + backstop/guía de prompt en S2/S3.

Decisión de producto (owner 2026-06-27): arroz/locrio/pasta en DESAYUNO = SIEMPRE duro;
"arroz de noche" / comida de desayuno en CENA = duro con DEGRADACIÓN a advisory en el intento
final (nunca cero-plan).

Cubre:
  - SSOT constants.slot_violations_for_meal_name (match word-boundary sobre el nombre + exclusiones)
  - constants.canonical_slot_key / build_meal_timing_rules
  - graph_orchestrator._detect_slot_appropriateness (productor del gate S1)
  - graph_orchestrator.slot_coherence_backstop_for_meal (backstop S2/S3)
  - parser-anchored: el gate cableado en review_plan_node + inyecciones en swap/chat-modify
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

import constants as c
import graph_orchestrator as go

# Dir backend derivado de un módulo de prod (robusto ante el rootdir de pytest).
_BACKEND = Path(go.__file__).resolve().parent


# ─────────────────────────────── SSOT (constants) ───────────────────────────────

@pytest.mark.parametrize("name,slot,expect_hard", [
    ("Arroz blanco con pollo guisado", "desayuno", True),
    ("Locrio de pollo", "desayuno", True),
    ("Moro de gandules con cerdo", "desayuno", True),
    ("Espaguetis con pollo", "desayuno", True),
    ("Lasaña de carne", "desayuno", True),
    ("Sancocho de res", "desayuno", True),
])
def test_breakfast_lunch_dishes_flagged_hard(name, slot, expect_hard):
    """Platos de almuerzo/cena en el desayuno = violación DURA (no degrada)."""
    v = c.slot_violations_for_meal_name(name, slot)
    assert v, f"{name!r} en {slot} debería flagear"
    assert any(x["hard"] for x in v) == expect_hard


@pytest.mark.parametrize("name", [
    "Panqueques de harina de arroz",   # el modificador 'de arroz' NO es un plato de arroz (G5)
    "Crepes de harina con fresas",
    "Mangú con huevos y queso frito",
    "Avena cocida con frutas y maní",
    "Tostada integral con pasta de maní",
    "Batido verde con avena",
])
def test_creative_breakfasts_not_flagged(name):
    """Platos creativos legítimos de desayuno (G5) NO deben flagearse — protege harina→panqueques."""
    assert c.slot_violations_for_meal_name(name, "desayuno") == [], name


@pytest.mark.parametrize("name", [
    "Pollo a la plancha con arroz blanco",   # arroz de noche
    "Locrio de pollo ligero",
    "Cereal con leche y guineo",
    "Panqueques con sirope",
])
def test_dinner_breakfast_or_rice_flagged_soft(name):
    """Arroz de noche / comida de desayuno en la cena = violación SOFT (degrada en intento final)."""
    v = c.slot_violations_for_meal_name(name, "cena")
    assert v, name
    assert not any(x["hard"] for x in v), f"{name} debería ser soft, no hard"


@pytest.mark.parametrize("name,slot", [
    ("Pescado al horno con ensalada y batata", "cena"),
    ("Tortilla de vegetales con casabe", "cena"),
    ("Arroz con habichuela y pollo", "almuerzo"),   # arroz SÍ va en almuerzo
    ("Locrio de cerdo", "almuerzo"),
    ("Arroz con leche", "merienda"),                # postre/merienda no está en el mapa
])
def test_appropriate_dishes_not_flagged(name, slot):
    assert c.slot_violations_for_meal_name(name, slot) == [], f"{name} en {slot}"


def test_canonical_slot_key():
    assert c.canonical_slot_key("Desayuno") == "desayuno"
    assert c.canonical_slot_key("CENA") == "cena"
    assert c.canonical_slot_key("Merienda AM") == "merienda"
    assert c.canonical_slot_key("Almuerzo") == "almuerzo"
    assert c.canonical_slot_key("xyz-no-existe") is None
    assert c.canonical_slot_key(None) is None


def test_build_meal_timing_rules():
    des = c.build_meal_timing_rules("Desayuno")
    cen = c.build_meal_timing_rules("Cena")
    assert "COHERENCIA DE HORARIO" in des and "arroz" in des.lower()
    assert "arroz de noche" in cen.lower()
    assert c.build_meal_timing_rules("slot-desconocido") == ""


# ─────────────────────── Detector S1 + backstop S2/S3 (graph) ───────────────────────

def _plan(*meals):
    return {"days": [{"day": "Lunes", "meals": list(meals)}]}


def test_detector_flags_hard_and_soft():
    """Detector de S1: desayuno-locrio = hard; cena-arroz = soft; almuerzo-arroz = OK."""
    issues = go._detect_slot_appropriateness(_plan(
        {"meal": "Desayuno", "name": "Locrio de pollo"},
        {"meal": "Almuerzo", "name": "Arroz con habichuela y pollo"},
        {"meal": "Cena", "name": "Pollo a la plancha con arroz blanco"},
        {"meal": "Merienda", "name": "Yogur con fresas"},
    )["days"])
    assert len(issues) == 2, issues
    assert any(i["hard"] and i["slot"] == "desayuno" for i in issues)
    assert any((not i["hard"]) and i["slot"] == "cena" for i in issues)


def test_detector_clean_plan_no_issues():
    issues = go._detect_slot_appropriateness(_plan(
        {"meal": "Desayuno", "name": "Mangú con huevos"},
        {"meal": "Almuerzo", "name": "Arroz, habichuela y pollo guisado"},
        {"meal": "Cena", "name": "Pescado al horno con vegetales"},
    )["days"])
    assert issues == []


def test_backstop_for_update_surfaces():
    """slot_coherence_backstop_for_meal (S2/S3): cena con arroz viola; cena coherente no; slot raro = []."""
    assert go.slot_coherence_backstop_for_meal({"name": "Pollo con arroz blanco"}, "Cena")
    assert go.slot_coherence_backstop_for_meal({"name": "Pescado al horno con ensalada"}, "Cena") == []
    assert go.slot_coherence_backstop_for_meal({"name": "Arroz"}, "slot-raro") == []
    assert go.slot_coherence_backstop_for_meal({}, "Cena") == []


def test_gate_default_on():
    assert go.SLOT_APPROPRIATENESS_GATE_ENABLED is True


# ──────────────────────────── Parser-anchored (no-regression) ────────────────────────────

def _src(name):
    return (_BACKEND / name).read_text(encoding="utf-8")


def test_anchor_present_in_all_surfaces():
    """tooltip-anchor presente en las 4 superficies — un renombre falla el test antes que prod."""
    for f in ("constants.py", "graph_orchestrator.py", "agent.py", "tools.py"):
        assert "P1-SLOT-APPROPRIATENESS" in _src(f), f"falta anchor en {f}"


def test_gate_wired_in_review_plan_node():
    """El gate S1 invoca el detector bajo el knob (no quedó código muerto)."""
    src = _src("graph_orchestrator.py")
    assert "SLOT_APPROPRIATENESS_GATE_ENABLED = _env_bool(" in src
    assert "_detect_slot_appropriateness(" in src
    # el detector se invoca DENTRO de review_plan_node (después de su def)
    idx_review = src.index("async def review_plan_node")
    assert src.index("_detect_slot_appropriateness(", idx_review) > idx_review


def test_swap_has_backstop_and_prompt_injection():
    src = _src("agent.py")
    assert "slot_coherence_backstop_for_meal(" in src
    assert "build_meal_timing_rules" in src
    assert "SLOT_INCOHERENCE" in src


def test_chat_modify_injects_timing_rules():
    src = _src("tools.py")
    assert "build_meal_timing_rules" in src
