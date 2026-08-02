# [P1-VOLUME-FALLBACK-DENSITY · 2026-08-02] El fallback del pantry guard para unidades
# irresolubles multiplicaba ×5 TAMBIÉN los mililitros y los gramos, inventando densidad
# 5 g/ml: "1 taza de Lechosa" (236.6 ml) se convertía en ~1183 g y "excedía matemáticamente"
# una lechosa entera. El LLM no puede corregir un exceso fantasma → agotaba retries → slot
# conservado. Casos reales del regen-day 2026-08-02 (plan 9d0cba11, corr=19cb565d):
#   - [1/2 cda Comino]  7.39 ml  → fallback viejo ~36.97 g > sobre de 28 g   → RECHAZO
#   - [1 taza Lechosa]  236.6 ml → fallback viejo ~1182.94 g > 1 unidad      → RECHAZO
#   - [1/2 taza Casabe] 118.3 ml → fallback viejo ~591.47 g > 0.62 lb (281g) → RECHAZO
# Con densidad agua (1 g/ml) los tres pasan. El ×5 queda SOLO para conteos imprecisos
# ("rebanada", "lonja", unidades raras) — su intención original.
#
# El bug era antiguo pero estaba enmascarado: el ledger plan-derived tenía denominadores
# enormes. P1-PANTRY-STRICT-CONSENT (Nevera real) los encogió y el ×5 explotó.
import re
from pathlib import Path

import pytest

import constants
from constants import validate_ingredients_against_pantry

_SRC = (Path(__file__).resolve().parent.parent / "constants.py").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# A) Parser: el anchor y la estructura del fallback existen
# ---------------------------------------------------------------------------

def test_anchor_present_in_source():
    assert "tooltip-anchor: P1-VOLUME-FALLBACK-DENSITY" in _SRC


def test_fallback_branches_ml_g_else():
    """El bloque fallback distingue ml (densidad agua) / g (identidad) / resto (×5).
    Parser-based: si alguien colapsa las ramas de vuelta al ×5 universal, esto falla
    antes de tocar producción."""
    m = re.search(
        r"if gen_base_unit == 'ml':\s*\n\s*fallback_g = gen_base_qty \* 1\.0\s*\n"
        r"\s*elif gen_base_unit == 'g':\s*\n\s*fallback_g = gen_base_qty\s*\n"
        r"\s*else:\s*\n\s*fallback_g = gen_base_qty \* 5\.0",
        _SRC,
    )
    assert m, "el fallback ml/g/else de P1-VOLUME-FALLBACK-DENSITY no está en constants.py"


# ---------------------------------------------------------------------------
# B) Funcional: los 3 casos reales de producción PASAN con el fix
#    (alimentos sin entry en VOLUMETRIC_DENSITIES para forzar el fallback)
# ---------------------------------------------------------------------------

def _guard(generated, pantry):
    return validate_ingredients_against_pantry(generated, pantry, strict_quantities=True)


def test_media_cda_comino_pasa_contra_sobre_28g():
    """7.39 ml de comino ≈ 7.4 g (agua) < 28 g del sobre. Pre-fix: 36.97 g → rechazo."""
    result = _guard(["1/2 cda Comino"], ["1 sobre (28 g) de Comino"])
    assert result is True, f"comino legítimo rechazado: {result}"


def test_taza_lechosa_pasa_contra_una_unidad():
    """236.6 ml de lechosa ≈ 236.6 g < 1 unidad (~1 kg). Pre-fix: 1182.94 g → rechazo."""
    result = _guard(["1 taza de Lechosa en cubos"], ["1 unidad de Lechosa"])
    assert result is True, f"taza de lechosa legítima rechazada: {result}"


def test_media_taza_casabe_pasa_contra_restos():
    """118.3 ml de casabe ≈ 118.3 g < 0.62 lb (281 g). Pre-fix: 591.47 g → rechazo."""
    result = _guard(["1/2 taza de Casabe (triturado)"], ["0.62 lb de Casabe"])
    assert result is True, f"media taza de casabe legítima rechazada: {result}"


def test_exceso_real_de_volumen_sigue_rechazando():
    """La honestidad no se pierde: 6 tazas (~1.4 L ≈ 1420 g) de lechosa SÍ exceden una
    unidad (~1 kg) incluso con densidad agua — el guard debe seguir rechazando."""
    result = _guard(["6 tazas de Lechosa en cubos"], ["1 unidad de Lechosa"])
    assert result is not True, "un exceso genuino de volumen dejó de rechazarse"


def test_conteo_impreciso_conserva_5g_por_unidad():
    """El ×5 sobrevive para conteos: 2 'rebanadas' de un alimento irresoluble ≈ 10 g.
    Contra 28 g disponibles debe pasar; contra 6 g debe rechazar."""
    assert _guard(["2 rebanadas de Casabe"], ["1 sobre (28 g) de Casabe"]) is True
    result = _guard(["2 rebanadas de Casabe"], ["1 sobre (6 g) de Casabe"])
    assert result is not True, "el fallback 5g/unidad para conteos se perdió"


# ---------------------------------------------------------------------------
# C) Marker bump
# ---------------------------------------------------------------------------

def test_last_known_pfix_bumped():
    """[de-pin pattern · 2026-08-02] Formato-only: no pineamos el literal (churn en cada
    P-fix posterior); el cross-link marker↔test lo enforza test_p2_hist_audit_14."""
    from app import _LAST_KNOWN_PFIX
    assert re.match(r"^P\d-[A-Z0-9-]+ · \d{4}-\d{2}-\d{2}$", _LAST_KNOWN_PFIX)
