"""[P2-AUDIT-2 · 2026-05-10] `_strip_dairy_brand` quita marcas comerciales
de lácteos para que el master_map del aggregator canonicalice al base
("Leche", "Yogurt", "Queso").

Bug original (audit 2026-05-10):
    Recetas con "Leche Induvaca entera" + lista con "Leche" → guard
    reportaba false positive `cap_swallowed_modifier`. El master_map
    no aliasa todas las marcas (explotaría la cardinalidad).

Fix:
    Helper `_strip_dairy_brand(name)` quita marca SI el nombre menciona
    producto lácteo (gate de keyword). Sin gate, "Rica salsa picante"
    perdería "rica" mal. Knob `MEALFIT_COHERENCE_DAIRY_BRANDS` para
    extensión runtime.

Cobertura:
    - Strip de marca cuando hay keyword lácteo.
    - NO strip si nombre no menciona lácteo (ej. "Rica salsa").
    - NO strip si no hay marca (ej. "Leche entera" → "Leche entera").
    - Multi-brand: "Pasteurizadora Rica" antes que "Rica" (sort len DESC).
    - Knob runtime añade marcas extras.
    - Wired en `_canonicalize_for_coherence`.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

import shopping_calculator
from shopping_calculator import _strip_dairy_brand, _canonicalize_for_coherence


# ---------------------------------------------------------------------------
# 0. Stub master_map para aislar el test (no requiere DB)
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def no_master_db(monkeypatch):
    monkeypatch.setattr(shopping_calculator, "get_master_ingredients", lambda: [])


# ---------------------------------------------------------------------------
# 1. Strip básico: marca + keyword lácteo
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,expected", [
    # Leche
    ("Leche Induvaca entera", "Leche entera"),
    ("Leche Rica deslactosada", "Leche deslactosada"),
    ("Leche parmalat semidescremada", "Leche semidescremada"),
    ("Leche Cofadel", "Leche"),
    # Yogurt
    ("Yogurt Rica griego", "Yogurt griego"),
    ("Yogur Yoplait natural", "Yogur natural"),
    ("Yoghurt Yogu fresa", "Yoghurt fresa"),
    # Queso
    ("Queso Sosúa rallado", "Queso rallado"),
    ("Queso sosua fresco", "Queso fresco"),
    # Crema / Mantequilla
    ("Crema Rica de leche", "Crema de leche"),
    ("Mantequilla Rica sin sal", "Mantequilla sin sal"),
])
def test_dairy_brand_stripped_when_keyword_present(name, expected):
    """Marca + keyword lácteo → strip. Case-preserve del resto."""
    assert _strip_dairy_brand(name) == expected, (
        f"P2-AUDIT-2 regresión: `_strip_dairy_brand({name!r})` debía "
        f"devolver {expected!r}. Si falla, master_map no encontrará "
        f"alias y el guard reportará false positive."
    )


# ---------------------------------------------------------------------------
# 2. Long brand antes que short brand (orden por len DESC)
# ---------------------------------------------------------------------------
def test_long_brand_takes_precedence_over_short():
    """'Pasteurizadora Rica' debe strippearse completo, no solo 'Rica'
    dejando 'Pasteurizadora' en el nombre."""
    result = _strip_dairy_brand("Leche Pasteurizadora Rica entera")
    # Debe quedar "Leche entera" (sin 'Pasteurizadora').
    assert result == "Leche entera", (
        f"P2-AUDIT-2 regresión: orden de strip incorrecto. "
        f"`Pasteurizadora Rica` debe matchear como unidad antes que `Rica`. "
        f"Got {result!r}."
    )


# ---------------------------------------------------------------------------
# 3. NO strip si nombre no menciona lácteo (evita falsos positivos)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", [
    "Rica salsa picante",       # "rica" es adjetivo, no marca
    "Sosúa fresco",              # "sosúa" sin lácteo → no toca
    "Salmón Induvaca",           # marca de lácteo aplicada a pescado (raro) → no toca
    "Pan",                       # sin marca, sin lácteo
    "Aceite Yoplait",            # marca lácteo + producto no-lácteo → no toca
])
def test_no_strip_without_dairy_keyword(name):
    """Sin keyword lácteo en el nombre, NO strippeamos — gate previene
    false positives sobre adjetivos/coincidencias."""
    assert _strip_dairy_brand(name) == name, (
        f"P2-AUDIT-2 regresión: `_strip_dairy_brand({name!r})` strippeó "
        f"sin keyword lácteo. Esto causa false positives sobre adjetivos "
        f"(ej. 'rica salsa' donde 'rica' es adjetivo, no marca)."
    )


# ---------------------------------------------------------------------------
# 4. NO strip si no hay marca (no-op)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", [
    "Leche",
    "Leche entera",
    "Yogurt natural",
    "Queso fresco",
    "Mantequilla sin sal",
])
def test_no_strip_without_brand(name):
    """Nombres lácteos sin marca permanecen intactos."""
    assert _strip_dairy_brand(name) == name, (
        f"P2-AUDIT-2 regresión: `_strip_dairy_brand({name!r})` mutó "
        f"sin marca presente. El no-op debe ser idempotente."
    )


# ---------------------------------------------------------------------------
# 5. Edge cases: None, empty, no-string
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", [None, "", "   "])
def test_empty_input_returns_input(name):
    """None / vacío → devolver el input sin tocar (no crash)."""
    assert _strip_dairy_brand(name) == name


def test_non_string_input_returns_as_is():
    """Inputs no-string (int, dict) → return as-is (defensivo)."""
    assert _strip_dairy_brand(123) == 123
    d = {"key": "val"}
    assert _strip_dairy_brand(d) is d


# ---------------------------------------------------------------------------
# 6. Knob runtime: MEALFIT_COHERENCE_DAIRY_BRANDS
# ---------------------------------------------------------------------------
def test_knob_extra_brands_added(monkeypatch):
    """Knob CSV añade marcas regionales/nuevas sin redeploy."""
    monkeypatch.setenv("MEALFIT_COHERENCE_DAIRY_BRANDS", "marca_x,marca_y")
    assert _strip_dairy_brand("Leche marca_x entera") == "Leche entera"
    assert _strip_dairy_brand("Queso marca_y rallado") == "Queso rallado"


def test_knob_empty_uses_defaults_only(monkeypatch):
    """Knob vacío → solo defaults. 'marca_x' no debe stripperarse."""
    monkeypatch.setenv("MEALFIT_COHERENCE_DAIRY_BRANDS", "")
    # 'marca_x' no es default → no toca.
    assert _strip_dairy_brand("Leche marca_x entera") == "Leche marca_x entera"
    # Default sigue funcionando.
    assert _strip_dairy_brand("Leche Induvaca entera") == "Leche entera"


# ---------------------------------------------------------------------------
# 7. Wired en _canonicalize_for_coherence
# ---------------------------------------------------------------------------
def test_canonicalize_for_coherence_strips_brand_before_master_lookup():
    """`_canonicalize_for_coherence` debe aplicar `_strip_dairy_brand` ANTES
    del lookup en master_map para que 'Leche Induvaca' → 'Leche'."""
    result = _canonicalize_for_coherence({"Leche Induvaca entera"})
    # master_map stub vacío → fallback: strip dairy → "Leche entera" →
    # _strip_trailing_modifier_es quita 'entera' → "Leche".
    assert "Leche" in result, (
        f"P2-AUDIT-2 regresión: 'Leche Induvaca entera' no canonicaliza "
        f"a 'Leche'. Got {result}. El helper no está wired al pipeline."
    )


def test_canonicalize_does_not_alter_non_dairy_with_brand_name():
    """`Rica salsa picante` (no lácteo) → NO strip 'rica'."""
    result = _canonicalize_for_coherence({"Rica salsa picante"})
    # Debe contener 'rica' o 'salsa' o similar — pero NO debe perder 'rica'
    # como si fuera marca.
    assert any("rica" in r.lower() for r in result), (
        f"P2-AUDIT-2 regresión: 'rica' (adjetivo) fue strippeado de "
        f"'Rica salsa picante'. Got {result}."
    )


# ---------------------------------------------------------------------------
# 8. Parser-based: knob registrado en _KNOBS_REGISTRY
# ---------------------------------------------------------------------------
def test_knob_registers_with_knobs_registry():
    """Knob debe llamarse a `_register_knob` para visibilidad en
    `/admin/knobs`."""
    _SC = Path(__file__).resolve().parent.parent / "shopping_calculator.py"
    src = _SC.read_text(encoding="utf-8")
    m = re.search(
        r"def\s+_get_extra_dairy_brands\s*\(\s*\)[^:]*:(.*?)(?=^def\s)",
        src,
        re.DOTALL | re.MULTILINE,
    )
    assert m, "P2-AUDIT-2: `_get_extra_dairy_brands` no encontrada."
    body = m.group(1)
    assert "MEALFIT_COHERENCE_DAIRY_BRANDS" in body, (
        "P2-AUDIT-2 regresión: knob name ya no referenciada en el helper."
    )
    assert "_register_knob" in body, (
        "P2-AUDIT-2 regresión: `_register_knob` no se llama. Sin esto, "
        "el knob no aparece en `/admin/knobs`."
    )


def test_strip_dairy_brand_wired_in_canonicalize_source():
    """Parser-based: `_canonicalize_for_coherence` llama a
    `_strip_dairy_brand(...)`. Sin esto, el helper existe pero no
    se aplica al pipeline."""
    _SC = Path(__file__).resolve().parent.parent / "shopping_calculator.py"
    src = _SC.read_text(encoding="utf-8")
    # Localizar la función _canonicalize_for_coherence.
    m = re.search(
        r"def\s+_canonicalize_for_coherence\s*\([^)]*\)[^:]*:(.*?)(?=^def\s)",
        src,
        re.DOTALL | re.MULTILINE,
    )
    assert m, "P2-AUDIT-2: `_canonicalize_for_coherence` no encontrada."
    body = m.group(1)
    assert "_strip_dairy_brand" in body, (
        "P2-AUDIT-2 regresión: `_strip_dairy_brand` no se llama en "
        "`_canonicalize_for_coherence`. El helper está definido pero "
        "no wired al pipeline — Leche Induvaca seguirá divergiendo."
    )
