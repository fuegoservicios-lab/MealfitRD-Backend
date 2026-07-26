"""[P1-COHERENCE-UNIT-MISMATCH-SYM · 2026-07-25] El guard ahogaba su propia señal.

Dos corridas seguidas se rechazaron por `COHERENCIA RECETAS LISTA`, cada una quemando una
regeneración COMPLETA (~150 s). Al mirar las divergencias reales del plan entregado `0bfe19ac`:

    total 40 divergencias · con delta FINITO: 3

Las otras 37 eran unidades incomparables, no defectos:

    Harina de trigo   receta {'cda': 0.73, 'taza': 0.49}   lista "1 paquete"   → delta = inf
    Orégano dominicano  receta {'cdta': …}                 lista "1 sobre"     → delta = inf
    Batata            receta {'unidad': …}                 lista "680 g"       → delta = inf

El alimento SÍ está en las recetas; buscar `paquete` entre las unidades de la receta da 0, y
`exp_qty == 0` se interpretaba como "fantasma en la lista". Las 3 comparaciones reales eran
Pulpo (+152%, envase mínimo), Limón (−44%, sub-oferta de verdad) y Sardinas (−18%, marginal).

## El fix es el espejo de uno que ya existía

`P2-COHERENCE-PACKAGE-UNITS` (2026-06-22) cerró la dirección `act_qty == 0` — la lista no tiene la
unidad pero sí el alimento — y la tagea `unit_mismatch`. La dirección simétrica (`exp_qty == 0`)
quedó abierta, emitiendo `hypothesis=unknown` + `delta_pct=inf`.

⚠️ Esa es la inestabilidad de conteos que `P1-REVIEW-COHERENCE-SEVERE-ONLY` y
`P1-COHERENCE-COUNT-MATERIAL` llevan meses conteniendo **aguas abajo**: el whack-a-mole de
"1 divergencia crítica sobre una proteína rotativa" se alimentaba de ruido que nunca debió
contarse. Arreglarlo en el origen es más barato que seguir subiendo umbrales.

**No debilita la detección**: un fantasma de verdad (alimento ausente de toda receta) tiene
`exp_units` vacío o a cero → `unit_mismatch` False → sigue siendo divergencia real. Eso está
anclado abajo, porque es la mitad que importa.
"""
import pytest

import shopping_calculator as sc


def _guard(expected, aggregated):
    """Invoca el comparador con los dos lados ya resueltos."""
    return sc._compare_expected_vs_aggregated(expected, aggregated, tolerance=0.10) \
        if hasattr(sc, "_compare_expected_vs_aggregated") else None


# ───────────── 1. el artefacto medido ─────────────

@pytest.mark.parametrize("food,exp_units,lista_unit,lista_qty", [
    ("Harina de trigo", {"cda": 0.73, "taza": 0.49}, "paquete", 1.0),
    ("Orégano dominicano", {"cdta": 3.5}, "sobre", 1.0),
    ("Batata", {"unidad": 1.5}, "g", 680.4),
])
def test_unidad_incomparable_se_tagea_no_se_reporta_como_fantasma(food, exp_units, lista_unit, lista_qty):
    divs = sc.compare_expected_vs_aggregated(
        {food: exp_units}, {food: {lista_unit: lista_qty}}, tolerance=0.10)
    d = next((x for x in divs if x.get("unit") == lista_unit), None)
    assert d is not None, divs
    assert d.get("unit_mismatch") is True, d


def test_fantasma_DE_VERDAD_sigue_siendo_divergencia():
    """La mitad que importa: un alimento que NINGUNA receta pide no puede quedar tapado."""
    divs = sc.compare_expected_vs_aggregated({}, {"Guanábana": {"unidad": 2.0}}, tolerance=0.10)
    d = next(x for x in divs if x["food"] == "Guanábana")
    assert d.get("unit_mismatch") is False, d


def test_receta_con_unidades_a_cero_tampoco_tapa():
    divs = sc.compare_expected_vs_aggregated(
        {"Sal": {"pizca": 0.0}}, {"Sal": {"paquete": 1.0}}, tolerance=0.10)
    d = next(x for x in divs if x["food"] == "Sal")
    assert d.get("unit_mismatch") is False, d


# ───────────── 2. las comparaciones REALES no se tocan ─────────────

@pytest.mark.parametrize("food,exp,act,esperado_delta", [
    ("Pulpo", {"g": 135.0}, {"g": 340.194}, 1.52),
    ("Limón", {"unidad": 5.39}, {"unidad": 3.0}, 0.443),
    ("Sardinas en lata", {"lata": 2.45}, {"lata": 2.0}, 0.184),
])
def test_mismas_unidades_siguen_midiendo_magnitud(food, exp, act, esperado_delta):
    divs = sc.compare_expected_vs_aggregated({food: exp}, {food: act}, tolerance=0.10)
    d = next(x for x in divs if x["food"] == food)
    assert d["delta_pct"] == pytest.approx(esperado_delta, rel=0.02)
    assert d.get("unit_mismatch") is False


def test_dentro_de_tolerancia_no_reporta():
    assert sc.compare_expected_vs_aggregated(
        {"Pollo": {"g": 100.0}}, {"Pollo": {"g": 105.0}}, tolerance=0.10) == []


# ───────────── 3. knob ─────────────

def test_knob_de_rollback(monkeypatch):
    monkeypatch.setattr(sc, "COHERENCE_UNIT_MISMATCH_SYM", False)
    divs = sc.compare_expected_vs_aggregated(
        {"Harina de trigo": {"cda": 0.73}}, {"Harina de trigo": {"paquete": 1.0}}, tolerance=0.10)
    assert divs[0].get("unit_mismatch") is False
