"""[P1-VARIETY-SAME-DAY-PROTEIN · 2026-06-27] La misma proteína principal (INCLUIDO el huevo) no debe
repetirse en 2+ comidas del MISMO día — pedido del owner (vio huevo en desayuno+cena). Se permite repetir
en DÍAS DISTINTOS.

Cubre: build_variety_report.same_day_protein_repeats + _variety_repeat_gate_issues (gate de retry).
"""
from __future__ import annotations

import graph_orchestrator as go


def _meal(name, slot="almuerzo", ings=None):
    return {"name": name, "meal": slot, "ingredients": ings or []}


def _plan(*days):
    return {"days": [{"day": i + 1, "meals": list(d)} for i, d in enumerate(days)]}


def test_egg_twice_same_day_is_flagged_and_gated():
    """Huevo en 2 comidas del mismo día → same_day_protein_repeats>0 y gate de rechazo."""
    plan = _plan([
        _meal("Batido Tropical con Claras de Huevo", "desayuno"),
        _meal("Pollo a la Plancha con Arroz", "almuerzo"),
        _meal("Manzana con Maní", "merienda"),
        _meal("Tortilla de Huevos Enteros con Ñame", "cena"),
    ])
    vr = go.build_variety_report(plan)
    assert vr["same_day_protein_repeats"] >= 1, vr
    issues = go._variety_repeat_gate_issues(vr)
    assert any("MISMA PROTEÍNA" in i for i in issues), issues


def test_heavy_protein_twice_same_day_is_gated():
    """Pollo en almuerzo + cena el mismo día → gate."""
    plan = _plan([
        _meal("Avena con Fresas", "desayuno"),
        _meal("Pollo Guisado con Arroz", "almuerzo"),
        _meal("Yogurt con Granola", "merienda"),
        _meal("Pechuga de Pollo al Horno con Yuca", "cena"),
    ])
    vr = go.build_variety_report(plan)
    assert vr["same_day_protein_repeats"] >= 1
    assert go._variety_repeat_gate_issues(vr)


def test_egg_once_per_day_is_clean():
    """Huevo una vez al día + proteínas distintas → sin same-day repeat."""
    plan = _plan([
        _meal("Revoltillo de Huevos con Vegetales", "desayuno"),
        _meal("Res Guisada con Arroz", "almuerzo"),
        _meal("Yogurt con Fresas", "merienda"),
        _meal("Pescado al Horno con Ensalada", "cena"),
    ])
    vr = go.build_variety_report(plan)
    assert vr["same_day_protein_repeats"] == 0, vr
    assert not any("MISMA PROTEÍNA" in i for i in go._variety_repeat_gate_issues(vr))


def test_same_protein_on_different_days_is_allowed():
    """Huevo el día 1 y el día 2 (una vez cada día) → permitido (no es same-day)."""
    plan = _plan(
        [_meal("Revoltillo de Huevos", "desayuno"), _meal("Pollo con Arroz", "almuerzo")],
        [_meal("Tortilla de Huevos", "desayuno"), _meal("Res con Yuca", "almuerzo")],
    )
    vr = go.build_variety_report(plan)
    assert vr["same_day_protein_repeats"] == 0, vr


def test_legumes_repeat_same_day_not_gated():
    """Habichuela en almuerzo + cena (la bandera dominicana) NO se gatea — excluida a propósito."""
    plan = _plan([
        _meal("Avena con Manzana", "desayuno"),
        _meal("Arroz con Habichuela Roja y Pollo", "almuerzo"),
        _meal("Fruta con Nueces", "merienda"),
        _meal("Habichuela Guisada con Res y Casabe", "cena"),
    ])
    vr = go.build_variety_report(plan)
    # pollo solo 1x, res solo 1x, habichuela excluida → 0 repeticiones de proteína gateada
    assert vr["same_day_protein_repeats"] == 0, vr


def test_knob_and_anchor_present():
    import re
    from pathlib import Path
    src = (Path(__file__).resolve().parent.parent / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert "P1-VARIETY-SAME-DAY-PROTEIN" in src
    assert "VARIETY_GATE_SAME_DAY_PROTEIN" in src
    assert "_SAME_DAY_PROTEIN_GATE_LABELS" in src
    # el set incluye huevo
    m = re.search(r"_SAME_DAY_PROTEIN_GATE_LABELS\s*=\s*.*huevo", src)
    assert m, "el set de proteínas gateadas same-day debe incluir 'huevo'"
