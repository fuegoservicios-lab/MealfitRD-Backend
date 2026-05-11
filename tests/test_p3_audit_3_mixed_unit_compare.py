"""[P3-AUDIT-3 · 2026-05-10] Tests para `compare_expected_vs_aggregated`
cubriendo el caso multi-unit en el mismo food.

Bug original (audit 2026-05-10):
    Tests existentes (`test_p1_shopping_recipe_coherence.py`) cubren
    `unit_mismatch` simple: receta en `cda`, lista en `unidad` (1 unit
    cada lado). NO cubren el caso de receta con MÚLTIPLES units del
    mismo food (ej. "1 lata + 1 cda de salsa picante") vs aggregator
    que puede colapsar mal o mantener separadas.

    Sin cobertura E2E del multi-unit, un refactor del aggregator que
    colapse unidades incorrectamente pasaría silencioso.

Fix:
    Tests funcionales que ejercen `compare_expected_vs_aggregated` con
    expected/aggregated multi-unit en el mismo food. Verifican:
      - Si AMBOS lados tienen las mismas units con qty match → no divergence.
      - Si una unit existe en expected pero NO en aggregated → divergencia
        en esa unit específica (no colapso silencioso).
      - Si aggregated tiene unit extra (fantasma multi-unit) → divergencia.
      - Magnitudes divergentes en una unit + match en otra → solo la
        unit divergente se reporta.
"""
from __future__ import annotations

import pytest

from shopping_calculator import compare_expected_vs_aggregated


# ---------------------------------------------------------------------------
# 1. Multi-unit match perfecto → sin divergencias
# ---------------------------------------------------------------------------
def test_multi_unit_perfect_match_no_divergence():
    """Receta "1 lata + 1 cda de salsa picante", lista igual → sin
    divergencias. Verifica que el comparador soporta multi-unit por
    food sin colapsar."""
    expected = {"Salsa Picante": {"lata": 1.0, "cda": 1.0}}
    aggregated = {"Salsa Picante": {"lata": 1.0, "cda": 1.0}}
    divergences = compare_expected_vs_aggregated(expected, aggregated, tolerance=0.05)
    assert divergences == [], (
        f"P3-AUDIT-3: multi-unit match perfecto debe dar 0 divergencias. "
        f"Got: {divergences}"
    )


# ---------------------------------------------------------------------------
# 2. Multi-unit con una unit faltante en aggregated → unit_mismatch
# ---------------------------------------------------------------------------
def test_multi_unit_one_unit_missing_in_aggregated():
    """Receta "1 lata + 1 cda", lista solo "1 lata" → divergencia
    SOLO en la unit faltante (`cda`), NO en la unit que matchea (`lata`)."""
    expected = {"Salsa Picante": {"lata": 1.0, "cda": 1.0}}
    aggregated = {"Salsa Picante": {"lata": 1.0}}  # falta cda
    divergences = compare_expected_vs_aggregated(expected, aggregated, tolerance=0.05)
    # Esperamos exactamente UNA divergencia en `cda`.
    units_reported = {d["unit"] for d in divergences}
    assert "cda" in units_reported, (
        f"P3-AUDIT-3 regresión: unit `cda` faltante en aggregated NO se "
        f"reportó. Divergences: {divergences}"
    )
    assert "lata" not in units_reported, (
        f"P3-AUDIT-3: unit `lata` matchea — no debe reportarse. "
        f"Divergences: {divergences}"
    )
    # Hipótesis debe ser cap_swallowed_modifier (qty=0 en aggregated para
    # esta unit pero el food existe en otra unit).
    cda_div = next(d for d in divergences if d["unit"] == "cda")
    assert cda_div["hypothesis"] in ("unit_mismatch", "cap_swallowed_modifier"), (
        f"P3-AUDIT-3: hipótesis inesperada para multi-unit miss. "
        f"Got: {cda_div['hypothesis']!r}"
    )


# ---------------------------------------------------------------------------
# 3. Multi-unit con unit extra en aggregated → fantasma multi-unit
# ---------------------------------------------------------------------------
def test_multi_unit_aggregated_has_extra_unit():
    """Receta "1 lata", lista "1 lata + 0.5 cda" → fantasma en `cda`
    (delta_pct = inf, expected=0, actual>0)."""
    expected = {"Salsa Picante": {"lata": 1.0}}
    aggregated = {"Salsa Picante": {"lata": 1.0, "cda": 0.5}}  # cda extra
    divergences = compare_expected_vs_aggregated(expected, aggregated, tolerance=0.05)
    units_reported = {d["unit"] for d in divergences}
    assert "cda" in units_reported, (
        f"P3-AUDIT-3: unit `cda` extra (fantasma multi-unit) NO se "
        f"reportó. Divergences: {divergences}"
    )
    # `lata` matchea → no reporta.
    assert "lata" not in units_reported
    # Fantasma → delta_pct=inf.
    cda_div = next(d for d in divergences if d["unit"] == "cda")
    assert cda_div["expected_qty"] == 0.0
    assert cda_div["actual_qty"] == 0.5
    assert cda_div["delta_pct"] == float("inf")


# ---------------------------------------------------------------------------
# 4. Multi-unit con magnitud divergente en una unit + match en otra
# ---------------------------------------------------------------------------
def test_multi_unit_magnitude_divergence_per_unit():
    """Receta "1 lata + 100 g", lista "1 lata + 20 g" → divergencia
    SOLO en `g` (80% delta > tolerance 5%), no en `lata` (match perfecto).
    Verifica que el comparador no colapsa magnitudes entre units.

    Nota: usamos ratio=0.2 (no 0.5) porque:
      - ratio=0.5 NO dispara pantry_overdeduct (check es STRICT `<`).
      - ratio=0.4 cae en banda yield_uncovered legume.
      - ratio=0.2 está fuera de yield bands Y bajo threshold default 0.5
        → pantry_overdeduct sin ambigüedad.
    """
    expected = {"Salsa": {"lata": 1.0, "g": 100.0}}
    aggregated = {"Salsa": {"lata": 1.0, "g": 20.0}}
    divergences = compare_expected_vs_aggregated(expected, aggregated, tolerance=0.05)
    units_reported = {d["unit"] for d in divergences}
    assert "g" in units_reported
    assert "lata" not in units_reported, (
        f"P3-AUDIT-3: unit `lata` matchea exacto — no debe reportarse. "
        f"Divergences: {divergences}"
    )
    g_div = next(d for d in divergences if d["unit"] == "g")
    assert g_div["expected_qty"] == 100.0
    assert g_div["actual_qty"] == 20.0
    # Magnitud → pantry_overdeduct (act < exp * 0.5, fuera de yield bands).
    assert g_div["hypothesis"] == "pantry_overdeduct", (
        f"P3-AUDIT-3: ratio 0.2 con default threshold debe ser "
        f"pantry_overdeduct. Got: {g_div['hypothesis']!r}"
    )


# ---------------------------------------------------------------------------
# 5. Multi-food multi-unit: divergencias por food independientes
# ---------------------------------------------------------------------------
def test_multi_food_multi_unit_independent():
    """2 foods cada uno con multi-unit. Divergencias por food no
    interfieren entre sí."""
    expected = {
        "Salsa": {"lata": 1.0, "cda": 2.0},
        "Aceite": {"l": 1.0, "cda": 5.0},
    }
    aggregated = {
        "Salsa": {"lata": 1.0, "cda": 1.0},  # cda mitad → divergencia
        "Aceite": {"l": 1.0, "cda": 5.0},     # match perfecto
    }
    divergences = compare_expected_vs_aggregated(expected, aggregated, tolerance=0.05)
    # Solo Salsa.cda divergente.
    foods_unit = {(d["food"], d["unit"]) for d in divergences}
    assert ("Salsa", "cda") in foods_unit
    assert ("Salsa", "lata") not in foods_unit  # match
    assert all(d["food"] != "Aceite" for d in divergences), (
        f"P3-AUDIT-3: Aceite matchea exacto — no debe aparecer. "
        f"Divergences: {divergences}"
    )


# ---------------------------------------------------------------------------
# 6. Mismo food, todas las units divergentes → todas reportadas
# ---------------------------------------------------------------------------
def test_multi_unit_all_divergent():
    """Si ambas units del mismo food divergen, ambas se reportan
    (no colapso oportunista). Usamos ratios 0.2 (fuera de yield bands)
    para clasificación determinística como pantry_overdeduct."""
    expected = {"Salsa Picante": {"lata": 5.0, "cda": 10.0}}
    aggregated = {"Salsa Picante": {"lata": 1.0, "cda": 2.0}}  # ambas ratio=0.2
    divergences = compare_expected_vs_aggregated(expected, aggregated, tolerance=0.05)
    units_reported = {d["unit"] for d in divergences}
    assert "lata" in units_reported, (
        f"P3-AUDIT-3: lata divergente NO se reportó. {divergences}"
    )
    assert "cda" in units_reported, (
        f"P3-AUDIT-3: cda divergente NO se reportó. {divergences}"
    )
    # Ambas son ratio 0.2 → pantry_overdeduct (< default 0.5, fuera de yield).
    for d in divergences:
        assert d["hypothesis"] == "pantry_overdeduct", (
            f"P3-AUDIT-3: ratio 0.2 debe ser pantry_overdeduct, "
            f"got {d['hypothesis']!r} for unit={d['unit']!r}"
        )
