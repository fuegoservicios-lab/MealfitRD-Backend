"""[P6-HTML-ENTITY-FIX] Tests para el decode de HTML entities en
ingredients antes de aggregation/consolidation.

Bug observable (PDF 2026-05-05 21:02:39 [775ce092]):
  Algunos correctores LLM emitieron HTML entities en JSON output:
    - '120g de mel&oacute;n' (en vez de '120g de melón')
    - 'Yaut&iacute;a blanca' (en vez de 'Yautía blanca')
    - 'Or&eacute;gano dominicano' (en vez de 'Orégano dominicano')

  Resultado:
    🛒 [AGGREGATE] 28 unique names: [..., 'Mel&oacute;n', 'Melón', ...]
    📦 [CONSOLIDATION] '120g de mel&oacute;n' -> '100 g de mel&oacute;n'

  El aggregator trataba 'Melón' y 'Mel&oacute;n' como ingredientes DIFERENTES.
  Lista de compras mostraba ambos como entradas separadas. Display al usuario
  veía 'Yaut&iacute;a blanca' literal. Caps no se acumulaban.

Fix:
  Aplicar `html.unescape()` en `_flatten_ingredient` — punto único por el que
  pasan TODOS los ingredients antes de cualquier procesamiento downstream.
  `html.unescape()` es idempotente: strings sin entities pasan inalterados.

Cobertura:
  - Las 5 vocales acentuadas (á/é/í/ó/ú) + ñ + ü
  - Repro casos del PDF
  - Idempotencia (Unicode ya correcto pasa inalterado)
  - Tipos no-string (lista, None) siguen funcionando
  - Mix entity+Unicode en mismo string
  - Symbol entities (&amp;, &lt;, &gt;) — defensiva
  - Sanity: marker en source
"""
import pytest


def _import_flatten():
    from graph_orchestrator import _flatten_ingredient
    return _flatten_ingredient


# ===========================================================================
# 1. Repro casos del PDF
# ===========================================================================
@pytest.mark.parametrize("raw,expected", [
    ("120g de mel&oacute;n", "120g de melón"),
    ("100g de mel&oacute;n", "100g de melón"),
    ("Yaut&iacute;a blanca", "Yautía blanca"),
    ("340g de yaut&iacute;a blanca", "340g de yautía blanca"),
    ("Or&eacute;gano dominicano", "Orégano dominicano"),
    ("1 cdta de or&eacute;gano dominicano", "1 cdta de orégano dominicano"),
])
def test_repro_pdf_html_entities_decoded(raw, expected):
    f = _import_flatten()
    assert f(raw) == expected, f"'{raw}' debe decodear a '{expected}'"


# ===========================================================================
# 2. Vocales acentuadas + ñ + ü (todos los entities relevantes en español)
# ===========================================================================
@pytest.mark.parametrize("entity,char", [
    ("&aacute;", "á"),
    ("&eacute;", "é"),
    ("&iacute;", "í"),
    ("&oacute;", "ó"),
    ("&uacute;", "ú"),
    ("&Aacute;", "Á"),
    ("&Eacute;", "É"),
    ("&Iacute;", "Í"),
    ("&Oacute;", "Ó"),
    ("&Uacute;", "Ú"),
    ("&ntilde;", "ñ"),
    ("&Ntilde;", "Ñ"),
    ("&uuml;", "ü"),
])
def test_spanish_entities_decoded(entity, char):
    f = _import_flatten()
    assert f(f"prefix{entity}suffix") == f"prefix{char}suffix"


# ===========================================================================
# 3. Idempotencia: strings sin entities pasan inalterados
# ===========================================================================
@pytest.mark.parametrize("text", [
    "Aguacate",
    "200g de pollo a la plancha",
    "Pechuga de pavo fresca",
    "Plátano verde con cilantro",  # Unicode ya correcto
    "Yautía blanca",
    "Orégano dominicano",
    "1 ½ tazas de arroz",  # Mixed con fracción Unicode
])
def test_unicode_correct_passes_through(text):
    f = _import_flatten()
    assert f(text) == text, f"'{text}' debe pasar inalterado"


# ===========================================================================
# 4. Mix de entities + Unicode en mismo string
# ===========================================================================
def test_mixed_entities_and_unicode_same_string():
    """Caso edge: el LLM puede emitir parte con entity, parte ya unicode."""
    f = _import_flatten()
    raw = "Mel&oacute;n con piña y or&eacute;gano"
    expected = "Melón con piña y orégano"
    assert f(raw) == expected


# ===========================================================================
# 5. Tipos no-string: lista, None
# ===========================================================================
def test_list_input_with_entities():
    """Lista anidada con entities — debe joinear y decodear."""
    f = _import_flatten()
    assert f(["200g", "de", "mel&oacute;n"]) == "200g de melón"


def test_none_input():
    f = _import_flatten()
    assert f(None) == ""


def test_empty_string():
    f = _import_flatten()
    assert f("") == ""


def test_int_input():
    """Robustez: si LLM devuelve un número (raro pero posible)."""
    f = _import_flatten()
    assert f(5) == "5"


# ===========================================================================
# 6. Symbol entities (defensiva)
# ===========================================================================
@pytest.mark.parametrize("entity,char", [
    ("&amp;", "&"),
    ("&lt;", "<"),
    ("&gt;", ">"),
    ("&quot;", '"'),
    ("&#39;", "'"),
])
def test_symbol_entities_decoded(entity, char):
    f = _import_flatten()
    assert f(f"a{entity}b") == f"a{char}b"


# ===========================================================================
# 7. Numeric entities (&#243; = ó)
# ===========================================================================
def test_numeric_entities_decoded():
    """Entities numéricas también deben decodear."""
    f = _import_flatten()
    assert f("Mel&#243;n") == "Melón"
    assert f("Yaut&#237;a") == "Yautía"


# ===========================================================================
# 8. NO regresión: strings que parecen entities pero NO lo son
# ===========================================================================
def test_non_entity_ampersand_preserved():
    """'A & B' NO debe ser tocado (no es entity válida)."""
    f = _import_flatten()
    assert f("Sal & Pimienta") == "Sal & Pimienta"


def test_partial_entity_string_preserved():
    """'&abc' sin punto y coma NO es entity válida — html.unescape lo deja."""
    f = _import_flatten()
    assert f("100g &abc def") == "100g &abc def"


# ===========================================================================
# 9. Sanity: marker en source
# ===========================================================================
def test_source_has_html_entity_fix_marker():
    import inspect
    import graph_orchestrator as go
    src = inspect.getsource(go._flatten_ingredient)
    assert "P6-HTML-ENTITY-FIX" in src, (
        "Marker debe existir; sin él alguien podría revertir el fix y "
        "reintroducir items duplicados Melón/Mel&oacute;n en aggregate"
    )
    assert "html.unescape" in src, "html.unescape debe usarse"


def test_idempotent_when_called_twice():
    """html.unescape es idempotente — aplicar 2 veces da mismo resultado."""
    f = _import_flatten()
    once = f("Mel&oacute;n")
    twice = f(once)
    assert once == twice == "Melón"
