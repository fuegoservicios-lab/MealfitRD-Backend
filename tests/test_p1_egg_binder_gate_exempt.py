"""[P1-EGG-BINDER-GATE-EXEMPT · 2026-07-11] El huevo-aglutinante no cuenta para el
gate same-day — en NINGUNA de las 4 superficies (paridad total).

Caso vivo corr=dbf45283 (renovación del owner, 15:00-15:12): el día 3 tenía 2+
platos-huevo donde los restantes eran AGLUTINANTES (croqueta/arepita: el huevo liga
la masa). El autofix los protege como funcionales (correcto: reescribir el huevo
rompe el plato) PERO el gate del revisor los contaba igual → rechazo INCORREGIBLE
por diseño (`egg_intrinsic_all_protected`) → 3 intentos quemados con banda 1.00 →
entrega degradada. La asimetría autofix-protege ↔ gate-cuenta ES el bug.

Contrato (las 4 superficies usan `_egg_counts_for_same_day_gate`, espejo EXACTO de
`_protected_binder`):
1. `build_variety_report.same_day_protein_repeats` (fuente del gate del revisor).
2. `_days_with_same_day_protein_repeat` (critique parity + telemetría REINTRO).
3. `_protein_repeat_autofix` day_hits (no reescribir un plato-huevo legítimo por
   culpa de una croqueta).
4. El binder con MODO DE COCCIÓN en el nombre ("Arepitas con Huevo Revuelto") SÍ
   cuenta — ahí el huevo es guarnición real, no ligante.

tooltip-anchor: P1-EGG-BINDER-GATE-EXEMPT
"""
from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BACKEND))


def _meal(slot, name, ings):
    return {"meal": slot, "name": name, "ingredients": list(ings),
            "ingredients_raw": list(ings), "recipe": ["Prepara."],
            "protein": 20, "carbs": 15, "fats": 8}


def _plan(meals):
    return {"days": [{"day": 1, "meals": meals}]}


def test_helper_mirrors_binder_protection():
    from graph_orchestrator import _egg_counts_for_same_day_gate
    assert _egg_counts_for_same_day_gate("revoltillo de huevo") is True
    assert _egg_counts_for_same_day_gate("tortilla de claras") is True
    assert _egg_counts_for_same_day_gate("croquetas de yuca") is False, "binder puro exento"
    assert _egg_counts_for_same_day_gate("arepitas de maiz") is False
    assert _egg_counts_for_same_day_gate("arepitas con huevo revuelto") is True, (
        "modo de cocción en el nombre = huevo-guarnición real → SÍ cuenta"
    )


def test_variety_report_exempts_binder():
    from graph_orchestrator import build_variety_report
    rep = build_variety_report(_plan([
        _meal("Desayuno", "Revoltillo de Huevo", ["2 huevos"]),
        _meal("Cena", "Croquetas de Yuca", ["1 huevo", "150 g de yuca"]),
    ]))
    assert int(rep.get("same_day_protein_repeats", 0)) == 0, (
        "revoltillo + croqueta same-day es LEGAL — contarlo produce rechazos "
        "incorregibles (corr=dbf45283: 3 intentos quemados con banda 1.00)"
    )


def test_variety_report_still_counts_real_repeats():
    from graph_orchestrator import build_variety_report
    rep = build_variety_report(_plan([
        _meal("Desayuno", "Revoltillo de Huevo", ["2 huevos"]),
        _meal("Cena", "Tortilla de Claras", ["4 claras de huevo"]),
    ]))
    assert int(rep.get("same_day_protein_repeats", 0)) >= 1, (
        "dos platos-huevo REALES same-day siguen contando (la exención es solo binder)"
    )


def test_critique_parity_helper_exempts_binder():
    from graph_orchestrator import _days_with_same_day_protein_repeat
    assert _days_with_same_day_protein_repeat(_plan([
        _meal("Desayuno", "Revoltillo de Huevo", ["2 huevos"]),
        _meal("Cena", "Croquetas de Yuca", ["1 huevo", "150 g de yuca"]),
    ])) == []
    assert _days_with_same_day_protein_repeat(_plan([
        _meal("Desayuno", "Revoltillo de Huevo", ["2 huevos"]),
        _meal("Cena", "Tortilla de Claras", ["4 claras"]),
    ])) == [1]


def test_side_modifier_binder_still_counts():
    from graph_orchestrator import build_variety_report
    rep = build_variety_report(_plan([
        _meal("Desayuno", "Revoltillo de Huevo", ["2 huevos"]),
        _meal("Cena", "Arepitas con Huevo Revuelto", ["2 huevos", "harina de maíz"]),
    ]))
    assert int(rep.get("same_day_protein_repeats", 0)) >= 1, (
        "'con Huevo Revuelto' = guarnición real de huevo → repite de verdad"
    )


def test_marker_anchored_in_source():
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert src.count("P1-EGG-BINDER-GATE-EXEMPT") >= 3
