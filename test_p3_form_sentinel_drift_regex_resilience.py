"""[P3-3 · 2026-05-07] Resilience tests del parser de `sentinels.js`.

Cierra P3-3 del backlog del audit `project_audit_p0_p1_close_2026_05_07.md`.

El gate de drift `test_p1_form_13_sentinel_drift.py` depende de un parser
basado en 2 regexes (`_SENTINELS_BLOCK_PATTERN`, `_KEY_VALUE_PATTERN`) que
opera sobre el texto crudo de `frontend/src/config/sentinels.js` (sin
intérprete JS). Si un futuro refactor introduce una variación de formato
que esos regexes no reconocen, el parser puede:

  (a) devolver dict vacío silenciosamente → `test_frontend_sentinels_subset...`
      pasa trivial (el `set()` vacío es subset de cualquier set). Rompe el
      gate sin levantar señal.
  (b) capturar pares espurios fuera del bloque SENTINELS → drift falso
      positivo.
  (c) fallar con AssertionError explícito → gate sigue funcionando.

Este archivo documenta el contrato del parser bajo ~12 inputs adversariales:
whitespace variations, comentarios, decoys textuales, unicode adversarial
(zero-width, RTL/LTR, homoglyphs), formato malformado.

Importamos los privates `_parse_sentinels_js`, `_SENTINELS_BLOCK_PATTERN`,
`_KEY_VALUE_PATTERN` desde el test P1-FORM-13 (SSOT).
"""
import importlib

import pytest


# Importar el módulo del gate P1-FORM-13 (su parser es el SSOT).
_p1_form_13 = importlib.import_module("test_p1_form_13_sentinel_drift")
_parse = _p1_form_13._parse_sentinels_js


_VALID_BODY_TEMPLATE = (
    "export const SENTINELS = Object.freeze({{\n"
    "    allergies: 'Ninguna',\n"
    "    medicalConditions: 'Ninguna',\n"
    "    dislikes: 'Ninguno',\n"
    "    struggles: 'Ninguno',\n"
    "}});\n"
)


# ---------------------------------------------------------------------------
# 1. Whitespace adversarial
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "snippet",
    [
        # tabs en lugar de espacios
        "export const SENTINELS = Object.freeze({\n"
        "\tallergies:\t'Ninguna',\n"
        "\tdislikes:\t'Ninguno',\n"
        "});",
        # todo en una sola línea
        "export const SENTINELS = Object.freeze({allergies:'Ninguna',dislikes:'Ninguno'});",
        # múltiples espacios alrededor del colon
        "export const SENTINELS = Object.freeze({\n"
        "    allergies     :     'Ninguna',\n"
        "    dislikes  :  'Ninguno',\n"
        "});",
        # blanks entre pares
        "export const SENTINELS = Object.freeze({\n\n\n"
        "    allergies: 'Ninguna',\n\n"
        "    dislikes: 'Ninguno',\n\n"
        "});",
    ],
    ids=["tabs", "single-line", "extra-spaces-around-colon", "extra-blank-lines"],
)
def test_parser_robust_to_whitespace_variations(snippet):
    parsed = _parse(snippet)
    assert parsed.get("allergies") == "Ninguna"
    assert parsed.get("dislikes") == "Ninguno"


# ---------------------------------------------------------------------------
# 2. Trailing comma optional
# ---------------------------------------------------------------------------
def test_parser_handles_no_trailing_comma():
    snippet = (
        "export const SENTINELS = Object.freeze({\n"
        "    allergies: 'Ninguna',\n"
        "    dislikes: 'Ninguno'\n"
        "});\n"
    )
    parsed = _parse(snippet)
    assert parsed == {"allergies": "Ninguna", "dislikes": "Ninguno"}


# ---------------------------------------------------------------------------
# 3. Comentarios JS dentro del bloque — limitación aceptada documentada
# ---------------------------------------------------------------------------
def test_parser_captures_decoy_inside_line_comment_inside_body():
    """LIMITACIÓN ACEPTADA: el parser NO strippea comentarios JS antes de
    aplicar `_KEY_VALUE_PATTERN`. Si alguien introduce
    `// example: 'fake'` dentro de SENTINELS, será capturado como un par.

    Defensa: si esto ocurriera, `test_parser_extracts_all_expected_field_keys`
    fallaría en P1-FORM-13 (extra fields). Este test documenta el bug-shape
    para que cualquier dev futuro que vea el ruido entienda la causa raíz.
    """
    snippet = (
        "export const SENTINELS = Object.freeze({\n"
        "    allergies: 'Ninguna',\n"
        "    // legacyKey: 'Sin alergia',  /* commented out */\n"
        "    dislikes: 'Ninguno',\n"
        "});\n"
    )
    parsed = _parse(snippet)
    assert parsed.get("allergies") == "Ninguna"
    assert parsed.get("dislikes") == "Ninguno"
    assert parsed.get("legacyKey") == "Sin alergia", (
        "Parser strippea comentarios? Si ahora lo hace y este assertion falla, "
        "documentar el cambio de contrato y considerar relajar el assertion."
    )


# ---------------------------------------------------------------------------
# 4. Decoy fuera del bloque NO debe contaminar
# ---------------------------------------------------------------------------
def test_parser_ignores_decoy_outside_sentinels_block():
    """Pares `key: 'value'` fuera del bloque `export const SENTINELS = ...`
    NO deben aparecer en el resultado. El bloque-scoped DOTALL del primer
    regex aísla la búsqueda."""
    snippet = (
        "// allergies: 'IGNORE_ME'\n"
        "const otherConfig = { allergies: 'IGNORE_OTHER' };\n"
        "export const SENTINELS = Object.freeze({\n"
        "    allergies: 'Ninguna',\n"
        "});\n"
        "export const HELPER = { dislikes: 'IGNORE_HELPER' };\n"
    )
    parsed = _parse(snippet)
    assert parsed == {"allergies": "Ninguna"}, (
        f"Parser capturó pares fuera del bloque: {parsed!r}"
    )


# ---------------------------------------------------------------------------
# 5. Bloque ausente → AssertionError explícito (no dict vacío silencioso)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "snippet",
    [
        "// archivo vacío\n",
        "export const SENTINELS = { allergies: 'Ninguna' };\n",  # sin Object.freeze
        "export const OTHER = Object.freeze({ allergies: 'Ninguna' });\n",  # nombre distinto
        "const SENTINELS = Object.freeze({ allergies: 'Ninguna' });\n",  # sin export
    ],
    ids=["empty", "no-freeze", "wrong-name", "no-export"],
)
def test_parser_raises_assertion_on_missing_block(snippet):
    with pytest.raises(AssertionError, match="No se encontró"):
        _parse(snippet)


# ---------------------------------------------------------------------------
# 6. Bloque presente pero body vacío → dict vacío (NO crash)
# ---------------------------------------------------------------------------
def test_parser_returns_empty_dict_when_body_is_empty():
    """Si el bloque existe pero está vacío (`{}`), el parser devuelve {}.
    El gate P1-FORM-13 entonces fallará en `test_parser_extracts_all_expected_field_keys`
    con missing fields. NO debe crashear el parser."""
    snippet = "export const SENTINELS = Object.freeze({});\n"
    parsed = _parse(snippet)
    assert parsed == {}


# ---------------------------------------------------------------------------
# 7. Quote style: el regex es estricto a single-quotes
# ---------------------------------------------------------------------------
def test_parser_misses_double_quoted_values_by_design():
    """`_KEY_VALUE_PATTERN` matchea sólo `'value'` (single quotes). Si alguien
    usa double quotes, el par NO es capturado.

    Por qué aceptar este comportamiento: ESLint del repo enforza single quotes
    en .js (convención frontend). Si pasara a double quotes, el gate
    `test_parser_extracts_all_expected_field_keys` fallaría con missing fields,
    lo cual es señal correcta de drift de convención."""
    snippet = (
        'export const SENTINELS = Object.freeze({\n'
        '    allergies: "Ninguna",\n'
        '    dislikes: \'Ninguno\',\n'
        '});\n'
    )
    parsed = _parse(snippet)
    assert "allergies" not in parsed, "Double quote inesperadamente capturado"
    assert parsed.get("dislikes") == "Ninguno"


# ---------------------------------------------------------------------------
# 8. Unicode adversarial: zero-width space dentro del valor
# ---------------------------------------------------------------------------
def test_parser_preserves_zero_width_space_inside_value():
    """Zero-width space (\\u200B) DENTRO del valor es invisible al ojo pero
    diferencia el string del backend SSOT. El parser debe preservarlo
    literalmente para que `test_frontend_sentinels_subset_of_backend_lowercased`
    detecte la divergencia silenciosa.

    Riesgo defensivo: un atacante (o copy-paste accidental desde Word/Slack)
    podría introducir zero-width chars que pasarían review visual."""
    zwsp = "​"
    snippet = (
        "export const SENTINELS = Object.freeze({\n"
        f"    allergies: 'Ningu{zwsp}na',\n"
        "});\n"
    )
    parsed = _parse(snippet)
    assert parsed.get("allergies") == f"Ningu{zwsp}na"
    # Lower-cased seguirá conteniendo el ZWSP, NO matcheará "ninguna" en backend.
    assert parsed["allergies"].lower() != "ninguna", (
        "Zero-width chars deben SOBREVIVIR al lowercase para que el drift gate"
        " pueda detectarlos."
    )


# ---------------------------------------------------------------------------
# 9. Unicode adversarial: homoglifo cyrillic 'а' (U+0430)
# ---------------------------------------------------------------------------
def test_parser_distinguishes_cyrillic_homoglyph_in_value():
    """La 'а' cyrillica (U+0430) es visualmente idéntica a la 'a' latina
    (U+0061) pero distinta en bytes. El parser DEBE preservar la cyrillica
    literalmente, y el lowercase no debe normalizarla — Python `.lower()`
    sobre cyrillic mantiene cyrillic. El drift gate detectará la divergencia."""
    cyrillic_a = "а"
    fake = f"Ningun{cyrillic_a}"  # visual: "Ninguna" (con 'a' cyrillic)
    snippet = (
        "export const SENTINELS = Object.freeze({\n"
        f"    allergies: '{fake}',\n"
        "});\n"
    )
    parsed = _parse(snippet)
    assert parsed["allergies"] == fake
    assert parsed["allergies"] != "Ninguna", (
        "El homoglyph cyrillic NO debe ser normalizado al latin"
    )


# ---------------------------------------------------------------------------
# 10. RTL marker dentro del valor (escenario raro pero documentado)
# ---------------------------------------------------------------------------
def test_parser_preserves_rtl_marker_inside_value():
    """El marker RTL (\\u200F) puede invertir el orden visual sin cambiar
    los bytes. Igual que zero-width: el parser preserva, el gate detecta."""
    rtl = "‏"
    snippet = (
        "export const SENTINELS = Object.freeze({\n"
        f"    allergies: 'Ninguna{rtl}',\n"
        "});\n"
    )
    parsed = _parse(snippet)
    assert parsed["allergies"].endswith(rtl)


# ---------------------------------------------------------------------------
# 11. JSDoc precediendo el bloque NO debe interferir
# ---------------------------------------------------------------------------
def test_parser_skips_jsdoc_above_block():
    snippet = (
        "/**\n"
        " * @typedef {Object} SentinelMap\n"
        " * @property {string} allergies - sentinel: 'IGNORE_JSDOC'\n"
        " */\n"
        "export const SENTINELS = Object.freeze({\n"
        "    allergies: 'Ninguna',\n"
        "});\n"
    )
    parsed = _parse(snippet)
    assert parsed == {"allergies": "Ninguna"}, (
        f"JSDoc encima del bloque contaminó el parse: {parsed!r}"
    )


# ---------------------------------------------------------------------------
# 12. Multiple SENTINELS-shaped blocks: matchea solo el primer match
# ---------------------------------------------------------------------------
def test_parser_picks_first_sentinels_block_when_multiple():
    """Si por error de refactor hay dos `export const SENTINELS = Object.freeze`,
    el parser captura el PRIMERO (re.search). Documentado para que ese
    escenario sea detectable con el gate de fields esperados — el segundo
    bloque sería ignored."""
    snippet = (
        "export const SENTINELS = Object.freeze({\n"
        "    allergies: 'PRIMERO',\n"
        "});\n"
        "export const SENTINELS = Object.freeze({\n"
        "    allergies: 'SEGUNDO',\n"
        "});\n"
    )
    parsed = _parse(snippet)
    assert parsed["allergies"] == "PRIMERO", (
        "Política `re.search` debe capturar el primer match"
    )


# ---------------------------------------------------------------------------
# 13. Drift end-to-end: parser captura adversarial → gate del backend lo rechaza
# ---------------------------------------------------------------------------
def test_e2e_drift_detection_under_zero_width_attack():
    """End-to-end: si el .js estuviera comprometido con un zero-width en
    'Ninguna', el gate del backend (test P1-FORM-13) detectaría el drift
    porque el `set` del frontend (con ZWSP, lowercased) no sería subset del
    `_SENTINEL_NONE_VALUES` del backend. Aquí simulamos el cálculo
    directamente para verificar que el parser+gate combinados rechazan el
    payload adversarial."""
    from graph_orchestrator import _SENTINEL_NONE_VALUES

    zwsp = "​"
    snippet = (
        "export const SENTINELS = Object.freeze({\n"
        f"    allergies: 'Ningu{zwsp}na',\n"
        "    medicalConditions: 'Ninguna',\n"
        "    dislikes: 'Ninguno',\n"
        "    struggles: 'Ninguno',\n"
        "});\n"
    )
    parsed = _parse(snippet)
    frontend_lower = {v.strip().lower() for v in parsed.values()}
    drift = frontend_lower - set(_SENTINEL_NONE_VALUES)
    assert drift, (
        "Drift adversarial NO detectado por el gate. Si este test falla, el "
        "backend está aceptando un sentinel con zero-width chars — investigar."
    )
    assert any(zwsp in d for d in drift), (
        "El elemento drift debe ser el que tiene zero-width — sanity del setup."
    )
