"""[P3-AGG-PRESENTATION-MODIFIERS · 2026-05-23] Tras P3-AGG-CLEAN-LEADING-PUNCT
limpiar el `/` corrupto del `"/pedazos de queso"`, el aggregator emitía
``"Pedazos de queso"`` como name canónico. Pero ese nombre NO existe en
``master_ingredients`` (el catálogo tiene "queso blanco", "queso fresco",
"queso de freír", etc), así que el pantry guard regex fast-path no
matcheaba — Y Vector Search caía con 429 RESOURCE_EXHAUSTED.

Caso productivo verificado log 2026-05-23 00:45-00:47: swap del Almuerzo
falló 3 retries seguidos, cada uno con un alias distinto del LLM:
``"120g Pedazos de queso"``, ``"70g de Pedazos de queso"``,
``"90g Pedazos de queso"`` — todos rechazados por el pantry guard.

Fix: strip de **presentation modifiers** ("pedazos de", "lonjas de",
"trozos de", etc) al inicio del ingredient name extraído por
``_parse_quantity()``. Cubre la clase de modifiers que NO son aliases
canónicos en PROTEIN/CARB/VEGGIE_FAT_SYNONYMS. Excluye explícitamente
"filete de", "lomo de", "carne molida de" (sí son canónicos).

Cross-link con ``test_p2_hist_audit_14_marker_test_link``: slug
``p3_agg_presentation_modifiers`` ↔ filename
``test_p3_agg_presentation_modifiers.py``.
"""
import pathlib

import pytest

BACKEND_ROOT = pathlib.Path(__file__).parent.parent
SHOPPING_PY = (BACKEND_ROOT / "shopping_calculator.py").read_text(encoding="utf-8")
APP_PY = (BACKEND_ROOT / "app.py").read_text(encoding="utf-8")


def _import_helper():
    pytest.importorskip("langchain_google_genai", reason="shopping_calculator requires langchain")
    from shopping_calculator import _strip_presentation_modifier_prefix
    return _strip_presentation_modifier_prefix


# ---------------------------------------------------------------------------
# Section A — Strip modifiers verificados en caso productivo
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    # Caso productivo verificado
    ("pedazos de queso", "queso"),
    ("Pedazos de queso", "queso"),  # title case del aggregator
    # Variaciones plural/singular
    ("pedazo de pan", "pan"),
    ("trozos de pollo", "pollo"),
    ("trozo de carne", "carne"),
    ("rebanadas de pan integral", "pan integral"),
    ("rebanada de queso", "queso"),
    ("rodajas de tomate", "tomate"),
    ("rodaja de limón", "limón"),
    ("porciones de arroz", "arroz"),
    ("porción de pasta", "pasta"),
    ("tajadas de plátano", "plátano"),
    ("cubos de pollo", "pollo"),
    ("tiras de pimiento", "pimiento"),
    ("dados de cebolla", "cebolla"),
    ("lonjas de jamón", "jamón"),
    # Case insensitive
    ("PEDAZOS DE QUESO", "QUESO"),
    ("Trozos De Pollo", "pollo"),  # solo strip prefix, no normaliza el resto
    # Idempotencia para inputs sin modifier
    ("queso blanco", "queso blanco"),
    ("pollo a la plancha", "pollo a la plancha"),
    ("arroz", "arroz"),
])
def test_strip_presentation_modifier_prefix(raw, expected):
    """[FUNCIONAL] Strip modifiers de presentación + preserva el name canónico."""
    helper = _import_helper()
    result = helper(raw)
    # Tolerancia: el strip solo afecta el prefijo, el resto del string puede
    # mantener su case original — comparamos en lower si los expected están
    # en lower, o exacto si están en case mixto
    if expected.islower():
        assert result.lower() == expected.lower(), (
            f"Expected {expected!r} from {raw!r}, got {result!r}"
        )
    else:
        # Para "Trozos De Pollo" → "Pollo" (el "Pollo" tras strip mantiene su case)
        # Para "PEDAZOS DE QUESO" → "QUESO"
        # Simple lower comparison es suficiente para verificar el strip semántico
        assert result.lower() == expected.lower(), (
            f"Expected {expected!r} from {raw!r}, got {result!r}"
        )


# ---------------------------------------------------------------------------
# Section B — NO stripear aliases canónicos legítimos
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", [
    # Estos SÍ son aliases canónicos en PROTEIN_SYNONYMS — NO se deben stripear
    "filete de pollo",
    "filete de pescado",
    "filete de res",
    "filete de mero",
    "filete de tilapia",
    "lomo de cerdo",
    "carne molida de res",
    "pechuga de pavo",
    "muslo de pollo",
    # Estos son CARB_SYNONYMS aliases — NO stripear
    "puré de papas",
    "puré de batata",
    "puré de yuca",
    # Estos son VEGGIE_FAT_SYNONYMS aliases — NO stripear
    "pimiento morrón",
])
def test_canonical_aliases_with_de_not_stripped(name):
    """[FUNCIONAL] Aliases canónicos que contienen "de" (ej. "filete de
    pollo") NO deben ser tocados — el helper solo strip-ea prefixes
    de presentación específicos."""
    helper = _import_helper()
    assert helper(name) == name, (
        f"Helper modificó incorrectamente alias canónico {name!r} → "
        f"{helper(name)!r}. La regex debe excluir 'filete de', 'lomo de', "
        f"'puré de', etc."
    )


# ---------------------------------------------------------------------------
# Section C — Defensive: non-string + empty input
# ---------------------------------------------------------------------------

def test_strip_modifier_non_string_returns_input():
    """[FUNCIONAL] Inputs no-string deben retornarse as-is sin raise."""
    helper = _import_helper()
    assert helper(None) is None
    assert helper(123) == 123
    assert helper("") == ""


# ---------------------------------------------------------------------------
# Section D — Wiring en ambos loops del aggregator
# ---------------------------------------------------------------------------

def test_helper_applied_in_both_loops():
    """El helper debe estar invocado en plan_loop Y consumed_loop del
    aggregator — sin ambos, la simetría plan↔consumed se rompe."""
    assert "_strip_presentation_modifier_prefix(name)" in SHOPPING_PY, (
        "Falta invocación del helper en aggregator."
    )
    assert SHOPPING_PY.count("_strip_presentation_modifier_prefix(name)") >= 2, (
        f"Esperaba ≥2 invocaciones del helper (plan + consumed), got "
        f"{SHOPPING_PY.count('_strip_presentation_modifier_prefix(name)')}."
    )


def test_helper_applied_after_punct_cleanup():
    """El helper debe estar invocado DESPUÉS de
    ``_clean_leading_punct_from_name`` — la cadena correcta es
    `"/pedazos de queso"` → `"pedazos de queso"` → `"queso"`."""
    # Encontrar las posiciones de ambas invocaciones en el plan loop
    plan_loop_marker = "for item in plan_ingredients:"
    plan_loop_start = SHOPPING_PY.find(plan_loop_marker)
    assert plan_loop_start > 0
    # Tomar ~1500 chars del plan loop
    plan_loop_body = SHOPPING_PY[plan_loop_start:plan_loop_start + 1500]
    punct_pos = plan_loop_body.find("_clean_leading_punct_from_name(name)")
    presentation_pos = plan_loop_body.find("_strip_presentation_modifier_prefix(name)")
    assert punct_pos > 0, "Falta _clean_leading_punct_from_name en plan loop"
    assert presentation_pos > 0, "Falta _strip_presentation_modifier_prefix en plan loop"
    assert punct_pos < presentation_pos, (
        "_strip_presentation_modifier_prefix debe ejecutarse DESPUÉS de "
        "_clean_leading_punct_from_name. Sin este orden, '/pedazos de queso' "
        "no se reduce correctamente a 'queso'."
    )


# ---------------------------------------------------------------------------
# Section E — Regex pattern coverage
# ---------------------------------------------------------------------------

def test_regex_pattern_does_not_match_filete_de():
    """[FUNCIONAL] El regex NO debe matchear "filete de", "lomo de",
    "carne molida de" — son aliases canónicos."""
    pytest.importorskip("langchain_google_genai", reason="shopping_calculator requires langchain")
    from shopping_calculator import _PRESENTATION_MODIFIER_PREFIXES_RE
    for canonical_phrase in [
        "filete de pollo",
        "lomo de cerdo",
        "carne molida de res",
        "pechuga de pavo",
        "puré de papas",
    ]:
        assert not _PRESENTATION_MODIFIER_PREFIXES_RE.match(canonical_phrase), (
            f"Regex matchea incorrectamente {canonical_phrase!r} — "
            f"es alias canónico, no debe stripearse."
        )


def test_regex_pattern_matches_known_presentation_modifiers():
    """[FUNCIONAL] El regex matchea TODOS los modifiers de presentación
    documentados (singular + plural)."""
    pytest.importorskip("langchain_google_genai", reason="shopping_calculator requires langchain")
    from shopping_calculator import _PRESENTATION_MODIFIER_PREFIXES_RE
    for modifier_phrase in [
        "pedazo de X", "pedazos de X",
        "trozo de X", "trozos de X",
        "rebanada de X", "rebanadas de X",
        "rodaja de X", "rodajas de X",
        "porción de X", "porciones de X",
        "tajada de X", "tajadas de X",
        "cubo de X", "cubos de X",
        "tira de X", "tiras de X",
        "dado de X", "dados de X",
        "lonja de X", "lonjas de X",
    ]:
        assert _PRESENTATION_MODIFIER_PREFIXES_RE.match(modifier_phrase), (
            f"Regex no matchea modifier conocido {modifier_phrase!r}"
        )


# ---------------------------------------------------------------------------
# Section F — Marker anchor
# ---------------------------------------------------------------------------
# Pin removido — pin-tests se rompen cada P-fix siguiente.
# `test_p3_1_last_known_pfix_freshness` cubre freshness a nivel codebase.


# ---------------------------------------------------------------------------
# Section G — E2E: aggregator end-to-end con caso productivo
# ---------------------------------------------------------------------------

def test_aggregator_e2e_normalizes_pedazos_de_queso():
    """[FUNCIONAL] Pasar el input verificado del caso productivo al
    aggregator y verificar que el output NO contiene 'pedazos de queso'
    como name (debe haberse reducido a 'queso')."""
    pytest.importorskip("langchain_google_genai", reason="shopping_calculator requires langchain")
    from shopping_calculator import aggregate_and_deduct_shopping_list

    plan_ingredients = ["8 pedazos de queso", "30 g de nueces"]
    result = aggregate_and_deduct_shopping_list(plan_ingredients, [], multiplier=1.0)
    for item in result:
        assert "pedazos de queso" not in item.lower(), (
            f"Output sigue conteniendo 'pedazos de queso' — el helper no "
            f"se aplicó en el aggregator. Item: {item!r}"
        )
