"""[P1-SAMEDAY-BURN-FIX · 2026-07-11] Menos intentos quemados por el gate same-day:
(a) condimentos-portadores ("caldo de pollo") NO cuentan como porción de proteína;
(b) re-autofix POST-chain — la última palabra la tiene el corrector, no el reintroductor.

Caso vivo (renovación corr=57a373e0): 2 de 3 intentos con banda 1.00 rechazados por
'atun'/'pollo' repetidos same-day en comidas cuyo NOMBRE no los menciona; la reparación
quirúrgica reprodujo el par exacto (determinista). Repro local confirmó DOS mecanismos:
1. "1/2 taza de caldo de pollo" en ingredientes cuenta como label 'pollo' para el gate
   (falso positivo culinario) y el autofix lo "curaba" reescribiendo el caldo a otra
   proteína ("caldo de pavo" — absurdo).
2. El autofix corre ANTES del chain de calidad (P0-BAND-PRE-REVIEW) y los re-closers del
   chain pueden reintroducir el repeat DESPUÉS — el warn REINTRO era profecía sin corrector.

Paridad gate↔critique↔autofix↔closer en el mismo commit (lección P1-CRITIQUE-SLOT-PARITY).
El plato NOMBRADO 'Caldo de Pollo' sigue contando (porción real, strip solo en ingredientes).

tooltip-anchor: P1-SAMEDAY-BURN-FIX
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BACKEND))


def _day(second_line, second_name="Queso de Freír en Salsa de Ají Morrón"):
    return [{"day": 1, "meals": [
        {"meal": "Desayuno", "name": "Revoltillo de Pechuga con Ají",
         "ingredients": ["120 g de pechuga de pollo", "1 ají cubanela"],
         "recipe": ["Cocina la pechuga."], "protein": 30, "carbs": 5, "fats": 8, "cals": 220},
        {"meal": "Cena", "name": second_name,
         "ingredients": ["100 g de queso de freír", second_line, "1 ají morrón"],
         "recipe": ["Prepara la salsa."], "protein": 25, "carbs": 6, "fats": 15, "cals": 260},
    ]}]


def test_condiment_carrier_not_counted_by_detector():
    import graph_orchestrator as go
    assert go._days_with_same_day_protein_repeat({"days": _day("1/2 taza de caldo de pollo")}) == [], (
        "caldo de pollo en INGREDIENTES no es porción de proteína — contarlo quemó 2 "
        "intentos con banda 1.00 (corr=57a373e0)"
    )


def test_condiment_carrier_not_counted_by_variety_report():
    import graph_orchestrator as go
    rep = go.build_variety_report({"days": _day("1/2 taza de caldo de pollo")})
    assert int(rep.get("same_day_protein_repeats", 0)) == 0, (
        "el gate real del reviewer (build_variety_report) debe compartir la exención"
    )


def test_dish_named_caldo_still_counts():
    import graph_orchestrator as go
    rep = go.build_variety_report(
        {"days": _day("1 papa", second_name="Caldo de Pollo con Víveres")})
    assert int(rep.get("same_day_protein_repeats", 0)) == 1, (
        "un plato LLAMADO 'Caldo de Pollo' es una porción real — el strip aplica solo a "
        "ingredientes, no al nombre"
    )


def test_autofix_leaves_broth_alone():
    import graph_orchestrator as go
    days = _day("1/2 taza de caldo de pollo")
    n = go._protein_repeat_autofix(days, {"allergies": []}, None)
    blob = " ".join(str(i) for i in days[0]["meals"][1]["ingredients"]).lower()
    assert "caldo de pollo" in blob, (
        "el autofix NO debe 'curar' reescribiendo el caldo a otra proteína "
        "('caldo de pavo' medido en repro local)"
    )
    assert n == 0


def test_real_filler_still_detected_and_fixed():
    import graph_orchestrator as go
    from nutrition_db import IngredientNutritionDB
    days = _day("40 g de pechuga de pollo")
    assert go._days_with_same_day_protein_repeat({"days": days}) == [1]
    n = go._protein_repeat_autofix(days, {"allergies": []}, IngredientNutritionDB())
    assert n >= 1
    assert go._days_with_same_day_protein_repeat({"days": days}) == [], (
        "el repeat REAL (línea de proteína filler) sigue detectándose y curándose"
    )


def test_closer_day_aware_labels_exclude_carrier():
    import graph_orchestrator as go
    labels = go._protein_gate_labels_in_meal(_day("1/2 taza de caldo de pollo")[0]["meals"][1])
    assert "pollo" not in labels


def test_accent_variants_covered():
    import graph_orchestrator as go
    assert go._strip_protein_condiment_carriers("1 consomé de pollo") == "1 caldo"
    assert go._strip_protein_condiment_carriers("1 cubito de caldo de res y 2 tazas") \
        == "1 cubito de caldo y 2 tazas"
    out = go._strip_protein_condiment_carriers("caldo de atún")
    assert "atun" not in out and "atún" not in out


def test_late_refix_wired_post_chain():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    i = src.find("_late_rep_days = _days_with_same_day_protein_repeat(result)")
    assert i != -1
    win = src[i:i + 1800]
    assert "SAMEDAY_LATE_REFIX_ENABLED" in win and "_protein_repeat_autofix(result.get(\"days\")" in win, (
        "el re-autofix POST-chain debe correr ANTES del warn REINTRO — sin él, el warn es "
        "profecía de rechazo sin corrector (2 intentos quemados)"
    )
    import graph_orchestrator as go
    assert go.SAMEDAY_LATE_REFIX_ENABLED is True, "corrector: default ON (rollback via knob)"


def test_parity_all_four_surfaces_anchored():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert src.count("_strip_protein_condiment_carriers(") >= 5, (
        "helper + 4 surfaces (detector SSOT, build_variety_report, labels_in_meal, "
        "autofix _alias_hit, blob del detalle REINTRO) — paridad en el mismo commit"
    )
    assert src.count("P1-SAMEDAY-BURN-FIX") >= 5
