"""[P3-PANTRY-GUARD-UNICODE-FRACTIONS · 2026-05-23] Tercer cleanup en
cascada del bucle de retries del swap-meal.

Tras P3-AGG-CLEAN-LEADING-PUNCT (limpio el `/`) y
P3-AGG-PRESENTATION-MODIFIERS (limpio "pedazos de"), el aggregator emite
`Queso blanco` como name canónico. Pero el pantry guard ahora rechaza
por `over_limit` porque el regex `_container_weight_re` no parsea las
fracciones Unicode (`½`, `¼`, `¾`, etc) que el aggregator emite en
display strings tipo ``"1 paquete (½ lb) de Queso blanco"``.

Caso productivo verificado log 2026-05-23 00:58-01:01: 3 retries fallidos
del swap-meal del Almuerzo, todos rechazados por:

```
RECHAZO | unauthorized=0 | over_limit=1
[100 g de queso blanco] (Pediste 100 g, convertido dinámicamente
 excede tu inventario de 1.0 unidad)
[65g de queso blanco] (Pediste 65 g, convertido excede 1.0 unidad)
[0.18 lb de Queso blanco] (Pediste 82 g, convertido excede 1.0 unidad)
```

El user tiene `1.0 paquete = 340g` (master_ingredients.container_weight_g)
+ `(½ lb) = 227g` embebido en el aggregator display string. Pero el
regex `_container_weight_re` no captura `½` porque solo aceptaba `\d+`.
→ cae al fallback ``UNIT_WEIGHTS`` que no tiene ``Queso blanco`` →
default conservador (~5g) → rechazo de cualquier pedido ≥10g.

## Fix

1. Extender ``_container_weight_re`` para capturar Unicode fractions
   (``½``, ``¼``, ``¾``, ``⅓``, ``⅔``, ``⅛``, ``⅜``, ``⅝``, ``⅞``,
   ``⅙``, ``⅚``).
2. Helper ``_parse_fraction_or_number(s)`` que convierte el grupo 1 del
   match: si es fraction Unicode lookup en map, sino ``float()``.
3. SQL one-shot: invalidar 4 caches ``aggregated_shopping_list*`` del
   plan productivo para que se recalculen con la nueva normalización.

Cross-link con ``test_p2_hist_audit_14_marker_test_link``: slug
``p3_pantry_guard_unicode_fractions`` ↔ filename
``test_p3_pantry_guard_unicode_fractions.py``.
"""
import pathlib

import pytest

BACKEND_ROOT = pathlib.Path(__file__).parent.parent
CONSTANTS_PY = (BACKEND_ROOT / "constants.py").read_text(encoding="utf-8")
APP_PY = (BACKEND_ROOT / "app.py").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Section A — Fraction map cubre fracciones comunes en es-DO
# ---------------------------------------------------------------------------

def test_unicode_fraction_map_present():
    """El map debe estar definido en el módulo constants."""
    assert "_UNICODE_FRACTION_MAP" in CONSTANTS_PY, (
        "Falta `_UNICODE_FRACTION_MAP` dict en constants.py. Sin él, "
        "el regex extendido captura el char pero no puede convertir."
    )


@pytest.mark.parametrize("char,expected_approx", [
    ("½", 0.5),
    ("¼", 0.25),
    ("¾", 0.75),
    ("⅓", 0.333),  # 1/3
    ("⅔", 0.667),  # 2/3
    ("⅛", 0.125),
    ("⅜", 0.375),
    ("⅝", 0.625),
    ("⅞", 0.875),
    ("⅙", 0.167),  # 1/6
    ("⅚", 0.833),  # 5/6
])
def test_fraction_map_values_match_decimal_equivalents(char, expected_approx):
    """Cada char Unicode del map debe convertirse al decimal correcto.
    Validación literal del map: extraemos el value del source vía
    pattern match minimal."""
    # Match: '<CHAR>': <NUMBER>  (acepta float literal o expresión como 1/3)
    import re as _re
    pat = _re.compile(
        rf'"{re.escape(char) if False else char}"\s*:\s*([0-9./]+)',
    )
    m = pat.search(CONSTANTS_PY)
    assert m, f"Fraction {char!r} no encontrada en map o sintaxis distinta"
    raw_value = m.group(1)
    # Evaluate as Python expression (acepta "0.5" o "1/3")
    try:
        actual = eval(raw_value)
    except Exception as e:
        pytest.fail(f"No pude evaluar el value del map para {char!r}: {raw_value!r} → {e}")
    assert abs(actual - expected_approx) < 0.01, (
        f"Fraction {char!r} → {actual} ≠ esperado {expected_approx}"
    )


import re


# ---------------------------------------------------------------------------
# Section B — Regex extendido captura fracciones Unicode
# ---------------------------------------------------------------------------

def test_regex_pattern_extended_for_unicode_fractions():
    """El regex ``_container_weight_re`` debe incluir los chars Unicode
    de fracciones en el grupo 1 del pattern."""
    # Match sobre la definición del regex
    pat = re.compile(
        r'_container_weight_re\s*=\s*re\.compile\(\s*[\'"]r?[\'"]?'
        r'(.*?)(?:\)\s*,|\),)',
        re.DOTALL,
    )
    m = re.search(r"_container_weight_re\s*=\s*re\.compile\((.*?)\)\s*\n", CONSTANTS_PY, re.DOTALL)
    assert m, "No se encontró la definición de _container_weight_re"
    regex_def = m.group(1)
    # Al menos ½, ¼, ¾, ⅓, ⅔ deben estar en el pattern
    for char in ["½", "¼", "¾", "⅓", "⅔"]:
        assert char in regex_def, (
            f"Fraction {char!r} ausente del regex pattern. Sin ella, "
            f"display strings tipo 'paquete ({char} lb)' no parsean."
        )


# ---------------------------------------------------------------------------
# Section C — Helper _parse_fraction_or_number maneja ambos formatos
# ---------------------------------------------------------------------------

def test_parse_fraction_helper_defined():
    """El helper debe estar definido inline en
    ``validate_ingredients_against_pantry`` (closure)."""
    assert "_parse_fraction_or_number" in CONSTANTS_PY, (
        "Falta helper `_parse_fraction_or_number` en constants.py."
    )


def test_callsite_uses_helper_not_raw_float():
    """El callsite del container_match group(1) debe usar el helper, NO
    ``float()`` directo. ``float('½')`` lanza ValueError → caída a
    fallback inseguro."""
    # Buscar el callsite específico de container_qty
    m = re.search(
        r"container_qty\s*=\s*_parse_fraction_or_number\(\s*container_match\.group\(1\)\s*\)",
        CONSTANTS_PY,
    )
    assert m, (
        "El callsite `container_qty = ...container_match.group(1)` debe "
        "usar `_parse_fraction_or_number()` en vez de `float()` directo."
    )


# ---------------------------------------------------------------------------
# Section D — End-to-end: validate_ingredients_against_pantry accepts
# pantry items with Unicode fractions
# ---------------------------------------------------------------------------

def test_validator_accepts_queso_blanco_with_fraction_in_paquete_display():
    """[FUNCIONAL] Caso productivo reproducido: pantry tiene
    ``"1 paquete (½ lb) de Queso blanco"`` (227g). LLM pide
    ``"100 g de queso blanco"`` (100g). El validator debe ACEPTAR
    (100g cabe en 227g). Pre-fix rechazaba por bug del fraction."""
    pytest.importorskip("langchain_google_genai", reason="constants requires langchain")
    from constants import validate_ingredients_against_pantry

    pantry = ["1 paquete (½ lb) de Queso blanco", "30 g de nueces"]
    generated = ["100 g de queso blanco", "20 g de nueces"]
    result = validate_ingredients_against_pantry(
        generated_ingredients=generated,
        pantry_ingredients=pantry,
        strict_quantities=True,
    )
    assert result is True, (
        f"Validator debió aceptar 100g de queso blanco vs ½ lb (227g) "
        f"en pantry. Got: {result!r}"
    )


def test_validator_still_rejects_overflow_after_fraction_fix():
    """[FUNCIONAL] El fix NO debe debilitar el guard — pedidos que
    GENUINAMENTE excedan el inventario deben seguir rechazándose."""
    pytest.importorskip("langchain_google_genai", reason="constants requires langchain")
    from constants import validate_ingredients_against_pantry

    pantry = ["1 paquete (½ lb) de Queso blanco"]  # 227g
    generated = ["500 g de queso blanco"]  # >227g — genuine overflow
    result = validate_ingredients_against_pantry(
        generated_ingredients=generated,
        pantry_ingredients=pantry,
        strict_quantities=True,
    )
    assert result is not True, (
        f"Validator debería rechazar 500g vs ½ lb (227g) inventory. "
        f"Got: {result!r}"
    )


# ---------------------------------------------------------------------------
# Section E — Marker bumped
# ---------------------------------------------------------------------------

def test_marker_bumped():
    # Pin removido — pin-tests se rompen cada P-fix siguiente.
    # `test_p3_1_last_known_pfix_freshness` cubre freshness a nivel codebase.
    pass
